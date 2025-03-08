"""
Configuration Module
------------------
This module loads and manages configuration settings for the RAG system.
Settings are loaded from environment variables with reasonable defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTORS_DIR = DATA_DIR / "vectors"

# Create directories if they don't exist
for dir_path in [DOCUMENTS_DIR, CHUNKS_DIR, VECTORS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Embedding settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Vector database settings
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(VECTORS_DIR))

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "development_secret_key")

# Retrieval settings
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7")) 