from dotenv import load_dotenv
load_dotenv()

import uuid
import shutil
from pathlib import Path

from fastapi import Body, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from helper import chat_gemini_llm, get_relevent_chunks, get_system_prompt, load_file, split_file, store_embeddings_vector_store
from embeddings_model import get_gemini_embeddings
from retriever import get_retriever

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
       
        #getting embedding model and store embedding on store
        gemini_embeddings = get_gemini_embeddings()
        store_embeddings_vector_store(gemini_embeddings, split_docs, user_id)

        print("Embedding stored successfully")

        return JSONResponse(
            status_code=200,
            content = {
                "status": "success",
                "filename": file.filename,
                "user_id" : user_id
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content = {
                "status": "failed",
                "content": str(e)
            }
        )


@app.post("/api/chat")
async def chat(payload: dict):
    try:
        user_id = payload.get("user_id")
        query = payload.get("query")

        #getting embedding model
        gemini_embeddings = get_gemini_embeddings()
        
        retriever = get_retriever(gemini_embeddings, user_id)

        relevant_chunks = get_relevent_chunks(retriever, query) 

        system_prompt = get_system_prompt(relevant_chunks)
        
        # Chat
        response = chat_gemini_llm(system_prompt, query)
        
        return JSONResponse(
            status_code=200,
            content = {
                "status": "success",
                "content": response.content
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content = {
                "status": "failed",
                "content": str(e)
            }
        )