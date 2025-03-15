"""
Vector Store Module
------------------
This module handles the storage and retrieval of vector embeddings.

Key components:
- Vector Store Interface: Abstract base class for vector stores
- Vector Store Implementations: Concrete implementation for Qdrant
- Persistence Mechanisms: Save and load vector stores
"""

from typing import Dict, Optional, Union

from rag.config import VECTOR_DB_PATH, VECTOR_DB_TYPE
from rag.utils import logger
from rag.vector_store.base import VectorStore
from rag.vector_store.qdrant_store import QdrantStore


def get_vector_store(
    vector_store_type: Optional[str] = None,
    collection_name: str = "rag_collection",
    persist_directory: Optional[str] = None,
) -> VectorStore:
    """
    Get a vector store instance.
    
    Args:
        vector_store_type: Type of vector store to use (only 'qdrant' is supported)
        collection_name: Name of the collection to use
        persist_directory: Directory to persist the vector store
        
    Returns:
        VectorStore: A vector store instance
    """
    # Use configured persist directory if not specified
    if persist_directory is None:
        persist_directory = VECTOR_DB_PATH
        
    # Debug info to help track issues
    logger.info(f"Creating vector store with collection_name={collection_name}")
    
    # Create and return a QdrantStore instance
    logger.info(f"Creating QdrantStore with collection_name={collection_name}")
    return QdrantStore(
        collection_name=collection_name,
        local_path=persist_directory,
    ) 