import os
from langchain_qdrant import QdrantVectorStore

retriever_cache = {}

# takes embedding as input and return the retriever
def get_retriever(embeddings, user_id: str):
    if user_id not in retriever_cache:
        retriever=QdrantVectorStore.from_existing_collection(
            url=os.getenv("VECTOR_DB_URI"),
            collection_name=f"{os.getenv('VECTOR_DB_COLLECTION_NAME')}_{user_id}",
            embedding=embeddings
        )
        retriever_cache[user_id] = retriever
        return retriever
    else:
        return retriever_cache[user_id]
         