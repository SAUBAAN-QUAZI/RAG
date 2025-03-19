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

# Base directories - Support both local and cloud deployments
# For Render, use their RENDER_ROOT environment variable as base
IS_RENDER = os.environ.get('RENDER', 'False').lower() == 'true'

if IS_RENDER:
    # On Render, use the provided environment path
    BASE_DIR = Path(os.environ.get('RENDER_ROOT', '/app'))
    # For persistent storage, use /tmp which is available during the lifetime of a process
    STORAGE_ROOT = Path(os.environ.get('STORAGE_ROOT', '/tmp'))
else:
    # Local development
    BASE_DIR = Path(__file__).resolve().parent.parent
    STORAGE_ROOT = BASE_DIR

# Data directories - configurable via environment variables
DATA_DIR = Path(os.environ.get('DATA_DIR', str(STORAGE_ROOT / "data")))
DOCUMENTS_DIR = Path(os.environ.get('DOCUMENTS_DIR', str(DATA_DIR / "documents")))
CHUNKS_DIR = Path(os.environ.get('CHUNKS_DIR', str(DATA_DIR / "chunks")))
VECTORS_DIR = Path(os.environ.get('VECTORS_DIR', str(DATA_DIR / "vectors")))

# Create directories if they don't exist
for dir_path in [DATA_DIR, DOCUMENTS_DIR, CHUNKS_DIR, VECTORS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
elif OPENAI_API_KEY == "your_openai_api_key":
    raise ValueError("Please replace the placeholder 'your_openai_api_key' with your actual OpenAI API key in the .env file")

# Print a debug message with a masked version of the API key
key_preview = OPENAI_API_KEY[:4] + "..." + OPENAI_API_KEY[-4:] if OPENAI_API_KEY else ""
print(f"Using OpenAI API key starting with: {key_preview}")

# Ragie API settings
USE_RAGIE = os.getenv("USE_RAGIE", "False").lower() == "true"
RAGIE_API_KEY = os.getenv("RAGIE_API_KEY")
RAGIE_WEBHOOK_SECRET = os.getenv("RAGIE_WEBHOOK_SECRET", "")

# If Ragie is enabled, verify API key is available
if USE_RAGIE:
    if not RAGIE_API_KEY:
        raise ValueError("Ragie is enabled, but RAGIE_API_KEY environment variable is not set")
    elif RAGIE_API_KEY == "your_ragie_api_key_here":
        raise ValueError("Please replace the placeholder 'your_ragie_api_key_here' with your actual Ragie API key in the .env file")
    
    # Print a debug message with a masked version of the API key
    ragie_key_preview = RAGIE_API_KEY[:4] + "..." + RAGIE_API_KEY[-4:] if RAGIE_API_KEY else ""
    print(f"Using Ragie API key starting with: {ragie_key_preview}")
    print("Ragie integration is ENABLED")
    
    # Log webhook status
    if RAGIE_WEBHOOK_SECRET:
        print("Ragie webhooks are configured with a secret")
    else:
        print("Ragie webhooks are not configured (no secret provided)")
else:
    print("Ragie integration is DISABLED")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Vector database settings
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma").lower()
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(VECTORS_DIR))
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")
VECTOR_DB_API_KEY = os.getenv("VECTOR_DB_API_KEY")

# Embedding model to use
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Retrieval settings
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# LLM response settings
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "1500"))

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "development_secret_key")

# CORS settings
ALLOW_CORS = os.getenv("ALLOW_CORS", "True").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://yourdomain.com").split(",") 