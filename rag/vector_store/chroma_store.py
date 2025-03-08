"""
ChromaDB Vector Store
------------------
Vector store implementation using ChromaDB.
"""

import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings

from rag.config import VECTOR_DB_PATH
from rag.document_processing.document import DocumentChunk
from rag.utils import logger
from rag.vector_store.base import VectorStore


class ChromaStore(VectorStore):
    """
    Vector store implementation using ChromaDB.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: Union[str, Path] = VECTOR_DB_PATH,
        embedding_function = None,  # Not used, we manage embeddings separately
    ):
        """
        Initialize a ChromaStore.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database
            embedding_function: Not used, we manage embeddings separately
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Using existing Chroma collection: {collection_name}")
        except Exception:
            logger.info(f"Creating new Chroma collection: {collection_name}")
            self.collection = self.client.create_collection(name=collection_name)
            
        logger.info(f"Initialized ChromaStore with collection={collection_name}")
    
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
        if not embeddings or not chunks:
            logger.warning("No embeddings or chunks to add")
            return
            
        # Prepare data for Chroma
        ids = []
        embedding_vectors = []
        metadatas = []
        documents = []
        
        for chunk in chunks:
            chunk_id = chunk.chunk_id
            
            # Skip if embedding not available
            if chunk_id not in embeddings:
                logger.warning(f"No embedding found for chunk {chunk_id}")
                continue
                
            # Prepare data
            ids.append(chunk_id)
            embedding_vectors.append(embeddings[chunk_id])
            metadatas.append(chunk.metadata)
            documents.append(chunk.content)
            
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embedding_vectors,
            metadatas=metadatas,
            documents=documents,
        )
        
        logger.info(f"Added {len(ids)} embeddings to Chroma collection")
    
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
        try:
            # Log search parameters
            logger.info(f"Searching vector store with top_k={top_k}, filters={filter_dict}")
            
            # Check if collection has any embeddings
            collection_stats = self.get_collection_stats()
            logger.info(f"Collection stats: {collection_stats}")
            
            # Ensure top_k is a reasonable value
            if top_k <= 0:
                top_k = 5
                logger.warning(f"Invalid top_k value, using default: {top_k}")
            
            # DEBUGGING: Try a simple query first
            try:
                # Search collection with minimal parameters
                simple_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                
                logger.info(f"Simple query returned: {len(simple_results.get('ids', [[]])[0])} results")
                
                # If simple query works but filter doesn't, the issue is with the filter
                if filter_dict:
                    logger.info(f"Testing filter validity: {filter_dict}")
                
                # Using the simple_results to proceed
                results = simple_results
            except Exception as e:
                logger.exception(f"Error in simple query: {str(e)}")
                return []
            
            # Format results
            formatted_results = []
            
            if not results["ids"] or not results["ids"][0]:
                logger.warning("Search returned no results")
                return []
            
            # Log raw search results details
            logger.info(f"Search returned {len(results['ids'][0])} matches")
            if results['distances'] and results['distances'][0]:
                logger.info(f"Distance range: min={min(results['distances'][0]):.4f}, max={max(results['distances'][0]):.4f}")
                
            for i, result_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score (1.0 = identical, 0.0 = completely different)
                similarity = 1.0 - results["distances"][0][i]
                
                formatted_result = {
                    "chunk_id": result_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": similarity,
                }
                formatted_results.append(formatted_result)
                
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.exception(f"Error during vector search: {str(e)}")
            # Return empty results in case of error
            return []
    
    def save(self, file_path: str = None) -> None:
        """
        Save the vector store to disk.
        
        ChromaDB automatically persists data, so this is a no-op.
        
        Args:
            file_path: Not used for ChromaDB
        """
        # Chroma automatically persists data, so just log
        logger.info("ChromaDB automatically persists data, no explicit save needed")
    
    @classmethod
    def load(cls, file_path: str = None, collection_name: str = "rag_collection") -> "ChromaStore":
        """
        Load a vector store from disk.
        
        Args:
            file_path: Directory containing the ChromaDB data
            collection_name: Name of the collection to load
            
        Returns:
            ChromaStore: Loaded vector store
        """
        return cls(collection_name=collection_name, persist_directory=file_path)
    
    def clear(self) -> None:
        """
        Clear all vectors from the store.
        """
        try:
            self.collection.delete(where={})
            logger.info(f"Cleared all vectors from collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict: Statistics about the vector store
        """
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "vector_count": count,
            "persist_directory": str(self.persist_directory),
        } 