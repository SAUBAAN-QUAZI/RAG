"""
Vector Store Factory
------------------
This module provides a factory for creating vector stores.
"""

from typing import Optional

from rag.config import VECTOR_DB_TYPE, VECTOR_DB_PATH
from rag.vector_store.base import VectorStore
from rag.vector_store.chroma_store import ChromaStore
from rag.vector_store.qdrant_store import QdrantStore
from rag.utils import logger


def create_vector_store(
    vector_db_type: Optional[str] = None,
    **kwargs
) -> VectorStore:
    """
    Create a vector store based on the specified type.
    
    Args:
        vector_db_type: Type of vector store to create ('chroma' or 'qdrant')
        **kwargs: Additional arguments to pass to the vector store constructor
        
    Returns:
        VectorStore: A vector store instance
    """
    # Use the configured vector_db_type if none provided
    db_type = vector_db_type or VECTOR_DB_TYPE
    
    # Debug logging to help identify issues
    logger.info(f"Requested vector store type: {db_type}")
    logger.info(f"Environment VECTOR_DB_TYPE: {VECTOR_DB_TYPE}")
    
    # Normalize to lowercase for more reliable comparison
    db_type = db_type.lower() if db_type else ""
    
    if db_type == "qdrant":
        # Create a Qdrant vector store
        logger.info("Creating Qdrant vector store")
        return QdrantStore(**kwargs)
    elif db_type == "chroma":
        # Create a Chroma vector store
        logger.info("Creating Chroma vector store")
        return ChromaStore(**kwargs)
    else:
        # Log warning for unknown type
        logger.warning(f"Unknown vector store type: '{db_type}', defaulting to chroma")
        return ChromaStore(**kwargs) 