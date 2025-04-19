from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import shutil
from fastapi import Body, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from helper import chat_llm, get_relevent_chunks, get_system_prompt, load_file, split_file, store_embeddings_vector_store
from embeddings_model import embeddings
from retriever import get_retriever
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/upload")
def upload_file(user_id: str = Body(...), file: UploadFile = File(...)):
    try:
        upload_dir = Path("files")
        upload_dir.mkdir(exist_ok=True)
       
        user_id = user_id or str(uuid.uuid4())
       
        # save this file
        file_path = upload_dir / f"{user_id}::{file.filename}"
       
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        docs = load_file(f"{user_id}::{file.filename}")
        split_docs = split_file(docs)
       
        #store embedding on store
        store_embeddings_vector_store(embeddings, split_docs, user_id)

        print("Embedding stored successfully")

        return {
            "status": "success", 
            "filename": file.filename,
            "user_id" : user_id
        }
    except Exception as e:
        print(e)
        return {"status": "error", "detail": str(e)}


@app.post("/api/chat")
async def chat(payload: dict):
    try:
        user_id = payload.get("user_id")
        query = payload.get("query")
        
        retriever = get_retriever(embeddings, user_id)

        relevant_chunks = get_relevent_chunks(retriever, query) 

        system_prompt = get_system_prompt(relevant_chunks)

        # Chat
        response = chat_llm(system_prompt, query)

        return {
            "status" : "success",
            "content" : response.content
        }
    except Exception as e:
        return {
            "status": "error", 
            "detail": str(e)
        }