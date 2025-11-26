import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import config as config

def build_vector_store(text_content):
    """Builds a FAISS vector store from text content using OpenAI Embeddings."""
    if not text_content.strip():
        print("No text content to build vector store. RAG will not be effective.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.create_documents([text_content])

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, openai_api_key=os.getenv("OPEN_AI_KEY"))
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store built successfully using OpenAI Embeddings.")
    return vector_store

def retrieve_context(query, vector_store, k=config.RETRIEVAL_K):
    """Retrieves relevant context from the vector store using MMR (Maximal Marginal Relevance)."""
    if vector_store is None:
        return "No knowledge base available."
    
    try:
        docs = vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=k*3,
            lambda_mult=0.5
        )
    except Exception as e:
        docs = vector_store.similarity_search(query, k=k)
    
    context = "\n".join([doc.page_content for doc in docs])
    return context

def save_vector_store(vector_store, path=None):
    """Save vector store to disk"""
    if path is None:
        path = config.VECTOR_STORE_PATH
    Path(path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(path)
    print(f"Vector store saved to {path}")

def load_vector_store(path=None):
    """Load vector store from disk"""
    if path is None:
        path = config.VECTOR_STORE_PATH
    
    if not os.path.exists(path):
        print(f"No vector store found at {path}")
        return None
        
    try:
        embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, openai_api_key=os.getenv("OPEN_AI_KEY"))
        vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        print(f"Vector store loaded from {path}")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None 
