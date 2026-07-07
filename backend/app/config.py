import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- API keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Storage paths ---
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
STORE_PATH = os.path.join(DATA_DIR, "store.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 700))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# --- Embeddings ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --- Retrieval ---
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", 10))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", 10))
TOP_K_FUSED = int(os.getenv("TOP_K_FUSED", 10))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", 4))
RRF_K = int(os.getenv("RRF_K", 60))

# --- Reranker ---
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")

# --- Collection ---
CHROMA_COLLECTION_NAME = "documents"
