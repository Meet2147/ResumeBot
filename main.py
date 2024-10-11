import os
import uuid
import json
import streamlit as st
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from pathlib import Path
from logger import get_logger
from PIL import Image
from byaldi import RAGMultiModalModel

# Initialize logger
logger = get_logger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploaded_documents'
SESSION_FOLDER = 'sessions'
INDEX_FOLDER = os.path.join(os.getcwd(), '.byaldi')
STATIC_FOLDER = os.path.join(os.getcwd(), 'static/images')

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Function to load RAG model for session
def load_rag_model_for_session(session_id):
    index_path = os.path.join(INDEX_FOLDER, session_id)
    if os.path.exists(index_path):
        try:
            RAG = RAGMultiModalModel.from_index(index_path)
            st.session_state['RAG_models'][session_id] = RAG
            logger.info(f"RAG model for session {session_id} loaded from index.")
        except Exception as e:
            logger.error(f"Error loading RAG model for session {session_id}: {e}")
    else:
        logger.warning(f"No index found for session {session_id}.")

# Function to save session data
def save_session_data(session_id, session_name, chat_history, indexed_files):
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    session_data = {
        'session_name': session_name,
        'chat_history': chat_history,
        'indexed_files': indexed_files
    }
    with open(session_file, 'w') as f:
        json.dump(session_data, f)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'RAG_models' not in st.session_state:
    st.session_state['RAG_models'] = {}
if 'indexer_model' not in st.session_state:
    st.session_state['indexer_model'] = 'vidore/colpali'  # Set default model
if 'generation_model' not in st.session_state:
    st.session_state['generation_model'] = 'qwen'  # Set default model
if 'resized_height' not in st.session_state:
    st.session_state['resized_height'] = 280  # Set default size
if 'resized_width' not in st.session_state:
    st.session_state['resized_width'] = 280  # Set default size

# Sidebar: Session Creation and Management
with st.sidebar:
    st.header("Session Management")

    # Check if the session ID exists; if not, prompt user to create a new session
    if st.session_state['session_id'] is None:
        st.session_state['session_id'] = str(uuid.uuid4())
        session_name = f"Session {len(os.listdir(SESSION_FOLDER)) + 1}"
        save_session_data(st.session_state['session_id'], session_name, [], [])
        st.session_state['session_name'] = session_name
        st.success(f"New session created: {st.session_state['session_id']}")

    st.write(f"Current session ID: {st.session_state['session_id']}")

    # Option to start a new session
    if st.button("Start New Session"):
        st.session_state['session_id'] = str(uuid.uuid4())
        session_name = f"Session {len(os.listdir(SESSION_FOLDER)) + 1}"
        st.session_state['session_name'] = session_name
        save_session_data(st.session_state['session_id'], session_name, [], [])
        st.success(f"New session started: {st.session_state['session_id']}")
    
    # Session switching option
    session_files = [f for f in os.listdir(SESSION_FOLDER) if f.endswith('.json')]
    if session_files:
        selected_session = st.selectbox("Switch Session", session_files)
        st.session_state['session_id'] = selected_session[:-5]  # Remove '.json'
        load_rag_model_for_session(st.session_state['session_id'])
        st.success(f"Switched to session: {st.session_state['session_id']}")

    # Settings for model and image configuration
    st.header("Settings")
    st.session_state['indexer_model'] = st.selectbox("Select Indexer Model", ["vidore/colpali", "vidore/colpali-v1.2", "vidore/colqwen2-v0.1"], index=["vidore/colpali", "vidore/colpali-v1.2", "vidore/colqwen2-v0.1"].index(st.session_state['indexer_model']))
    st.session_state['generation_model'] = st.selectbox("Select Generation Model", ["gemini", "qwen", "gpt4", "llama-vision", "pixtral", "molmo", "groq-llama-vision"], index=["gemini", "qwen", "gpt4", "llama-vision", "pixtral", "molmo", "groq-llama-vision"].index(st.session_state['generation_model']))
    st.session_state['resized_height'] = st.number_input("Image Resized Height (multiple of 28):", min_value=28, step=28, value=st.session_state['resized_height'])
    st.session_state['resized_width'] = st.number_input("Image Resized Width (multiple of 28):", min_value=28, step=28, value=st.session_state['resized_width'])

# Main Content: Document Uploads and Chat
st.header(f"Session ID: {st.session_state['session_id']}")

# Upload section (No indexing at this stage)
uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)
if uploaded_files:
    session_folder = os.path.join(UPLOAD_FOLDER, st.session_state['session_id'])
    os.makedirs(session_folder, exist_ok=True)
    indexed_files = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(session_folder, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())
        st.success(f"File {uploaded_file.name} uploaded successfully.")
        indexed_files.append(uploaded_file.name)

    # Save the files but do not index them yet
    st.session_state['uploaded_files'] = indexed_files
    st.success("Files uploaded. Ready for indexing.")

# Indexing Section
if 'uploaded_files' in st.session_state and st.session_state['uploaded_files']:
    if st.button("Start Indexing"):
        try:
            index_name = st.session_state['session_id']
            index_path = os.path.join(INDEX_FOLDER, index_name)
            RAG = index_documents(session_folder, index_name=index_name, index_path=index_path)
            st.session_state['RAG_models'][st.session_state['session_id']] = RAG
            save_session_data(st.session_state['session_id'], st.session_state['session_name'], st.session_state['chat_history'], st.session_state['uploaded_files'])
            st.success("Documents indexed successfully.")
        except Exception as e:
            st.error(f"Error indexing documents: {e}")

# Chat area
st.write("### Chat History")
for msg in st.session_state['chat_history']:
    if msg['role'] == 'user':
        st.write(f"**User**: {msg['content']}")
    else:
        st.write(f"**Assistant**: {msg['content']}")

# User query input
user_query = st.chat_input("Enter your query:")

if user_query:
    try:
        rag_model = st.session_state['RAG_models'].get(st.session_state['session_id'])
        if rag_model is None:
            st.error(f"RAG model not found for session {st.session_state['session_id']}")
        else:
            retrieved_documents = retrieve_documents(rag_model, user_query, st.session_state['session_id'])
            response = generate_response(retrieved_documents, user_query, st.session_state['session_id'], 
                                            st.session_state['resized_height'], st.session_state['resized_width'], 
                                            st.session_state['generation_model'])
            
            # Update chat history
            st.session_state['chat_history'].append({"role": "user", "content": user_query})
            st.session_state['chat_history'].append({"role": "assistant", "content": response})

            # Display response
            st.write("**Assistant**:", response)

            # Display images from the response
            if retrieved_documents:
                # Create columns for horizontal layout
                num_images = len(retrieved_documents)
                cols = st.columns(num_images)  # Create one column per image

                for idx, image_path in enumerate(retrieved_documents):
                    full_image_path = os.path.join(STATIC_FOLDER, image_path.split('/')[-2], image_path.split('/')[-1])
                    logger.info(f"Checking image path: {full_image_path}")
                    if os.path.exists(full_image_path):
                        # Resize the image to 280x280
                        img = Image.open(full_image_path)
                        img_resized = img.resize((280, 280))
                        with cols[idx]:  # Place each image in its own column
                            st.image(img_resized, caption=f"Retrieved Image: {os.path.basename(full_image_path)}")
                    else:
                        with cols[idx]:
                            st.warning(f"Image not found: {image_path}")

    except Exception as e:
        st.error(f"Error generating response: {e}")