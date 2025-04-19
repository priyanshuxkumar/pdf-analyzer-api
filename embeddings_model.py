from dotenv import load_dotenv
load_dotenv()

import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings


if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Failed to load Gemini API key")

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Failed to load OpenAI API key")


_gemini_embeddings = None
_openai_embeddings = None


# Creating OPENAI embedding model
openai_embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"), 
    model="text-embedding-3-large"
)

def get_openai_embeddings():
    global _openai_embeddings
    if _openai_embeddings is None:
        _openai_embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"), 
            model="text-embedding-3-large"
        )
        return _openai_embeddings
    else:
        return _openai_embeddings     


# Creating Gemini embedding model
def get_gemini_embeddings():
    global _gemini_embeddings
    if _gemini_embeddings is None:
        _gemini_embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv("GEMINI_API_KEY"),    
            model="models/text-embedding-004",
        )
        return _gemini_embeddings
    else: 
        return _gemini_embeddings     

