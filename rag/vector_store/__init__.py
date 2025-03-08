"""
Vector Store Module
------------------
This module handles the storage and retrieval of vector embeddings.

Key components:
- Vector Store Interface: Abstract base class for vector stores
- Vector Store Implementations: Concrete implementations for different backends
- Persistence Mechanisms: Save and load vector stores
"""

from typing import Dict, Optional, Union

from rag.config import VECTOR_DB_PATH, VECTOR_DB_TYPE
from rag.utils import logger
from rag.vector_store.base import VectorStore
from rag.vector_store.chroma_store import ChromaStore


def get_vector_store(
    vector_store_type: Optional[str] = None,
    collection_name: str = "rag_collection",
    persist_directory: Optional[str] = None,
) -> VectorStore:
    """
    Get a vector store instance.
    
    Args:
        vector_store_type: Type of vector store to use ('chroma' or 'faiss')
        collection_name: Name of the collection to use
        persist_directory: Directory to persist the database
        
    Returns:
        VectorStore: A vector store instance
        
    Raises:
        ValueError: If the vector store type is not supported
    """
    # Use configured vector store type if not specified
    if vector_store_type is None:
        vector_store_type = VECTOR_DB_TYPE
        
    # Use configured persist directory if not specified
    if persist_directory is None:
        persist_directory = VECTOR_DB_PATH
        
    vector_store_type = vector_store_type.lower()
    
    if vector_store_type == "chroma":
        logger.info(f"Creating ChromaStore with collection_name={collection_name}")
        return ChromaStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}") 