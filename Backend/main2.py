import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends, Response
from fastapi.middleware.sessions import SessionMiddleware
from fastapi.responses import JSONResponse
from typing import List
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from pathlib import Path
from logger import get_logger
from byaldi import RAGMultiModalModel

# Initialize FastAPI app
app = FastAPI()

# Enable session middleware for session management
app.add_middleware(SessionMiddleware, secret_key="Meet@21092")

# Logger setup
logger = get_logger(__name__)

# Configure directories
UPLOAD_FOLDER = 'uploaded_documents'
SESSION_FOLDER = 'sessions'
INDEX_FOLDER = os.path.join(os.getcwd(), '.byaldi')

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)

# Dictionary to store RAG models per session
RAG_models = {}

# Helper function to load RAG model for a session
def load_rag_model_for_session(session_id: str):
    index_path = os.path.join(INDEX_FOLDER, session_id)
    if os.path.exists(index_path):
        try:
            RAG = RAGMultiModalModel.from_index(index_path)
            RAG_models[session_id] = RAG
            logger.info(f"RAG model for session {session_id} loaded from index.")
        except Exception as e:
            logger.error(f"Error loading RAG model for session {session_id}: {e}")
    else:
        logger.warning(f"No index found for session {session_id}.")

# Helper function to load all existing indexes at startup
def load_existing_indexes():
    if os.path.exists(INDEX_FOLDER):
        for session_id in os.listdir(INDEX_FOLDER):
            if os.path.isdir(os.path.join(INDEX_FOLDER, session_id)):
                load_rag_model_for_session(session_id)
    else:
        logger.warning("No .byaldi folder found. No existing indexes to load.")

# Load indexes on app startup
@app.on_event("startup")
def initialize_app():
    load_existing_indexes()
    logger.info("Application initialized and indexes loaded.")

# Endpoint to create a new session
@app.post("/new_session")
async def new_session(response: Response):
    session_id = str(uuid.uuid4())
    session_files = os.listdir(SESSION_FOLDER)
    session_number = len([f for f in session_files if f.endswith('.json')]) + 1
    session_name = f"Session {session_number}"
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")

    session_data = {
        'session_name': session_name,
        'chat_history': [],
        'indexed_files': []
    }

    with open(session_file, 'w') as f:
        json.dump(session_data, f)

    response.set_cookie(key="session_id", value=session_id)
    logger.info(f"New session created: {session_id}")
    return JSONResponse(content={"message": "New session started.", "session_id": session_id})

# Endpoint to upload files and index them
@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...), request: Request = Depends()):
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session not found.")
    
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)
    uploaded_files = []
    
    for file in files:
        file_path = os.path.join(session_folder, file.filename)
        with open(file_path, 'wb') as f:
            f.write(await file.read())
        uploaded_files.append(file.filename)
        logger.info(f"File saved: {file_path}")
    
    # Indexing files
    if uploaded_files:
        try:
            index_name = session_id
            index_path = os.path.join(INDEX_FOLDER, index_name)
            indexer_model = request.cookies.get('indexer_model', 'vidore/colpali')
            RAG = index_documents(session_folder, index_name=index_name, index_path=index_path, indexer_model=indexer_model)
            if RAG is None:
                raise ValueError("Indexing failed: RAG model is None.")
            RAG_models[session_id] = RAG
            
            session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
            with open(session_file, 'r') as f:
                session_data = json.load(f)

            session_data['indexed_files'].extend(uploaded_files)
            with open(session_file, 'w') as f:
                json.dump(session_data, f)
                
            logger.info("Documents indexed successfully.")
            return JSONResponse(content={"success": True, "message": "Files indexed successfully.", "indexed_files": uploaded_files})
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return JSONResponse(content={"success": False, "message": f"Error indexing files: {str(e)}"})
    else:
        return JSONResponse(content={"success": False, "message": "No files were uploaded."})

# Endpoint to handle querying and generating responses
@app.post("/generate_response")
async def generate_response_endpoint(query: str = Form(...), request: Request = Depends()):
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session not found.")
    
    try:
        generation_model = request.cookies.get('generation_model', 'qwen')
        resized_height = int(request.cookies.get('resized_height', 280))
        resized_width = int(request.cookies.get('resized_width', 280))
        
        # Retrieve relevant documents
        rag_model = RAG_models.get(session_id)
        if rag_model is None:
            logger.error(f"RAG model not found for session {session_id}")
            return JSONResponse(content={"success": False, "message": "RAG model not found for this session."})
        
        retrieved_images = retrieve_documents(rag_model, query, session_id)
        logger.info(f"Retrieved images: {retrieved_images}")
        
        # Generate response
        response = generate_response(retrieved_images, query, session_id, resized_height, resized_width, generation_model)
        return JSONResponse(content={"success": True, "response": response, "images": retrieved_images})
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return JSONResponse(content={"success": False, "message": f"Error generating response: {e}"})

# Endpoint to switch between sessions
@app.post("/switch_session/{session_id}")
async def switch_session(session_id: str, response: Response):
    response.set_cookie(key="session_id", value=session_id)
    if session_id not in RAG_models:
        load_rag_model_for_session(session_id)
    logger.info(f"Switched to session: {session_id}")
    return JSONResponse(content={"message": "Switched session", "session_id": session_id})

# Endpoint to rename a session
@app.post("/rename_session")
async def rename_session(session_id: str = Form(...), new_session_name: str = Form(...)):
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        session_data['session_name'] = new_session_name
        with open(session_file, 'w') as f:
            json.dump(session_data, f)
        
        return JSONResponse(content={"success": True, "message": "Session name updated."})
    else:
        return JSONResponse(content={"success": False, "message": "Session not found."})

# Endpoint to delete a session
@app.post("/delete_session/{session_id}")
async def delete_session(session_id: str):
    try:
        session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
        
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        if os.path.exists(session_folder):
            import shutil
            shutil.rmtree(session_folder)
        
        RAG_models.pop(session_id, None)
        
        logger.info(f"Session {session_id} deleted.")
        return JSONResponse(content={"success": True, "message": "Session deleted successfully."})
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return JSONResponse(content={"success": False, "message": f"Error deleting session: {e}"})

# Endpoint to get indexed files for a session
@app.get("/get_indexed_files/{session_id}")
async def get_indexed_files(session_id: str):
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            indexed_files = session_data.get('indexed_files', [])
        return JSONResponse(content={"success": True, "indexed_files": indexed_files})
    else:
        return JSONResponse(content={"success": False, "message": "Session not found."})
