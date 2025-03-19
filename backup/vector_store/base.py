"""
Vector Store Base
-------------
This module defines the base interface for vector stores.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from rag.document_processing.document import DocumentChunk


class VectorStore(ABC):
    """
    Abstract base class for vector stores.
    """
    
    @abstractmethod
    def add_embeddings(
        self,
        embeddings: Dict[str, List[float]],
        chunks: List[DocumentChunk],
    ) -> None:
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: Dictionary mapping chunk IDs to embedding vectors
            chunks: List of document chunks
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search for similar vectors in the store.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List of dictionaries containing chunk information and similarity scores
        """
        pass
    
    @abstractmethod
    def save(self, file_path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            file_path: Path to save the vector store
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, file_path: str) -> "VectorStore":
        """
        Load a vector store from disk.
        
        Args:
            file_path: Path to the saved vector store
            
        Returns:
            VectorStore: Loaded vector store
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all vectors from the store.
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict: Statistics about the vector store
        """
        pass
    
    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists in the vector store.
        
        Args:
            document_id: ID of the document to check
            
        Returns:
            bool: True if the document exists in the vector store, False otherwise
        """
        pass 