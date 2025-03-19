"""
Vector Store Factory
------------------
This module provides a factory for creating vector stores.
"""

from typing import Optional

from rag.config import VECTOR_DB_PATH
from rag.vector_store.base import VectorStore
from rag.vector_store.qdrant_store import QdrantStore
from rag.utils import logger


def create_vector_store(
    vector_db_type: Optional[str] = None,
    **kwargs
) -> VectorStore:
    """
    Create a vector store based on the specified type.
    
    Args:
        vector_db_type: Type of vector store to create (only 'qdrant' supported)
        **kwargs: Additional arguments to pass to the vector store constructor
        
    Returns:
        VectorStore: A vector store instance
    """
    # Create a Qdrant vector store (only supported option)
    logger.info("Creating Qdrant vector store")
    return QdrantStore(**kwargs) 