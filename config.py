# ✅ UPDATED CONFIG FOR OPENAI
import os
from dotenv import load_dotenv

load_dotenv()

# File paths
TEXT_FILE_FOR_KNOWLEDGE = "data/text.txt"
BOOK_FILE_FOR_KNOWLEDGE = "data/extracted_book_text.txt"
BOOK2_FILE_FOR_KNOWLEDGE = "data/extracted_book2_text.txt"
BOOK3_FILE_FOR_KNOWLEDGE = "data/extracted_book3_text.txt"

# ✅ Updated model settings for OpenAI
LLM_MODEL = "gpt-4o"  # or "gpt-3.5-turbo" for a cheaper option
LLM_TEMPERATURE = 0.1
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vector store settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 3

# ✅ OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vector store path
VECTOR_STORE_PATH = "data/faiss_index"
