import os
import uuid
import json
import time
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from logger import get_logger
from byaldi import RAGMultiModalModel
import re
from pathlib import Path
# Initialize FastAPI app
app = FastAPI()

# Enable session middleware
app.add_middleware(SessionMiddleware, secret_key="Meet@21092")

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Logger setup
logger = get_logger(__name__)

# Configuration
UPLOAD_FOLDER = "uploaded_documents"
INDEX_FOLDER = os.path.join(os.getcwd(), '.byaldi')
SESSION_FOLDER = "sessions"

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)

# Cache for RAG models
RAG_models = {}


def secure_filename(filename: str) -> str:
    """
    Replace unwanted characters in a file name to prevent security issues.
    """
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)  # Allow only alphanumeric characters, dots, underscores, and dashes
    return filename
# Helper to load RAG model for a session
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
        logger.error(f"Index path {index_path} does not exist.")

# Routes

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        # Sanitize filename
        filename = secure_filename(file.filename)
        
        # Define the file path where the file will be saved
        file_location = Path(UPLOAD_FOLDER) / filename
        
        # Save the file
        with open(file_location, "wb") as f:
            f.write(await file.read())

        return {"info": f"File '{filename}' uploaded successfully."}
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Error uploading file")

@app.get("/new_session")
async def new_session(request: Request):
    session_id = str(uuid.uuid4())
    request.session['session_id'] = session_id
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

    return {"info": "New session started", "session_id": session_id}

@app.get("/get_indexed_files/{session_id}")
async def get_indexed_files(session_id: str):
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        indexed_files = session_data.get('indexed_files', [])
        return {"success": True, "indexed_files": indexed_files}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/index_documents")
async def index_documents_endpoint(session_id: str):
    load_rag_model_for_session(session_id)
    index_documents(session_id)
    return {"info": f"Documents indexed for session {session_id}"}

@app.post("/generate_response")
async def generate_response_endpoint(request: Request):
    session_id = request.session.get('session_id')
    if session_id not in RAG_models:
        load_rag_model_for_session(session_id)
    response = generate_response(session_id)
    return {"response": response}

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5050)
