from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

# take the filename as input and load the whole file
def load_file(file_name):
    pdf_path = Path(__file__).parent / "files" / file_name
    loader = PyPDFLoader(pdf_path)
    docs = loader.load() 
    return docs

# take the loaded file as input make the chunks of it
def split_file(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    return split_docs


# Store embedding in vector store (Takes input embeddings and split docs)
def store_embeddings_vector_store(embeddings, split_docs, user_id):
    try:
        if not split_docs:
            print("⚠️ split_docs is empty! Nothing to store.")
            return
        
        if not embeddings:
            print("⚠️ Embedding not provided.")
            return

        print('Injection Start')

        QdrantVectorStore.from_documents(
            documents=split_docs,
            url=os.getenv("VECTOR_DB_URI") or "http://localhost:6333",
            collection_name=f"{os.getenv('VECTOR_DB_COLLECTION_NAME')}_{user_id}",
            embedding=embeddings
        )

        print('Injection Done!')
    except Exception as e: 
        print("ERROR in storing vector store", str(e))
        raise e;   


# take retriever and query as input and return the relevent chunks
def get_relevent_chunks(retriever, query):
    relevant_chunks = retriever.similarity_search(query)
    return relevant_chunks


def get_system_prompt(relevent_chunks):
    return f"""
        You are an expert AI agent who is specialized in reading, summarize and explaning.
        You have no access of external data. You are not able to answer anything without the context.
        If you get the context (provided below) that is extracted from user documents. You can answer very efficiently.
        Output should be detailed 
       
        Rules:
            - Carefully analyze the user query. 
            - If the user query response is not found the clearly say : "The provided document doesnot contain any information about your query".


        Document Context: { relevent_chunks }

        Example:
        Input: What is defer keyword in the provided document.
        Output: A defer keyword meaning is execution of a function until the surrounding function returns.
    """

def chat_openai_llm(system_prompt, query):
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5, 
            max_tokens=100 
        )

        messages = [
            ("system", system_prompt),
            ("human", query)
        ]
        response = llm.invoke(messages)
        return response
    except Exception as e:
        raise e

def chat_gemini_llm(system_prompt, query):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=2, 
        )
        messages = [
            ("system", system_prompt),
            ("human", query),
        ]
        response = llm.invoke(messages)

        return response
    except Exception as e:
        raise e
        
    
    