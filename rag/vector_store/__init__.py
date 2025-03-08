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
from rag.vector_store.qdrant_store import QdrantStore


def get_vector_store(
    vector_store_type: Optional[str] = None,
    collection_name: str = "rag_collection",
    persist_directory: Optional[str] = None,
) -> VectorStore:
    """
    Get a vector store instance.
    
    Args:
        vector_store_type: Type of vector store to use ('chroma' or 'qdrant')
        collection_name: Name of the collection to use
        persist_directory: Directory to persist the database (for Chroma)
        
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
        
    # Debug info to help track issues
    logger.info(f"Creating vector store with type={vector_store_type}, "
                f"collection_name={collection_name}")
        
    # Normalize to lowercase for comparison
    vector_store_type = vector_store_type.lower() if vector_store_type else ""
    
    if vector_store_type == "chroma":
        logger.info(f"Creating ChromaStore with collection_name={collection_name}")
        return ChromaStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    elif vector_store_type == "qdrant":
        logger.info(f"Creating QdrantStore with collection_name={collection_name}")
        return QdrantStore(
            collection_name=collection_name,
            local_path=persist_directory,
        )
    else:
        logger.warning(f"Vector store type '{vector_store_type}' not recognized. "
                      f"Supported types are 'chroma' and 'qdrant'. Defaulting to 'chroma'.")
        return ChromaStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        ) 