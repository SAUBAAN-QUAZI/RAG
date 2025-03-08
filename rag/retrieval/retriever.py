"""
Retriever Module
------------
This module coordinates embedding and search to retrieve relevant documents.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from rag.config import (
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
    VECTORS_DIR,
    VECTOR_DB_TYPE
)
from rag.document_processing.document import DocumentChunk
from rag.embedding.service import EmbeddingService
from rag.utils import logger
from rag.vector_store import get_vector_store


class Retriever:
    """
    Retrieves relevant documents for a query using vector similarity search.
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_type: Optional[str] = None,
        collection_name: str = "rag_collection",
        persist_directory: Union[str, Path] = VECTORS_DIR,
        top_k: int = TOP_K_RESULTS,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        """
        Initialize a Retriever.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store_type: Type of vector store to use (defaults to config VECTOR_DB_TYPE)
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the vector store
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score to include results
        """
        # Create embedding service if not provided
        if embedding_service is None:
            self.embedding_service = EmbeddingService()
        else:
            self.embedding_service = embedding_service
        
        # Use config value if no vector_store_type provided    
        vector_store_type = vector_store_type or VECTOR_DB_TYPE
        logger.info(f"Initializing vector store of type: {vector_store_type}")
            
        # Create vector store
        self.vector_store = get_vector_store(
            vector_store_type=vector_store_type,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"Initialized Retriever with top_k={top_k}, "
                   f"similarity_threshold={similarity_threshold}")
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the retrieval system.
        
        Args:
            chunks: List of document chunks to add
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
            
        logger.info(f"Adding {len(chunks)} chunks to retrieval system")
        
        # Generate embeddings for chunks
        embeddings = self.embedding_service.embed_chunks(chunks)
        
        # Add embeddings to vector store
        self.vector_store.add_embeddings(embeddings, chunks)
        
        logger.info(f"Added {len(chunks)} chunks to retrieval system")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return (overrides instance setting)
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List of dictionaries containing retrieved chunks and similarity scores
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        # Use instance top_k if not specified
        if top_k is None:
            top_k = self.top_k
            
        # Generate embedding for query
        query_embedding = self.embedding_service.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
        )
        
        # Always log the raw results for debugging
        logger.info(f"Retrieved {len(results)} raw results from vector store")
        
        # DEBUGGING: Print out the first result's content preview if available
        if results and len(results) > 0:
            first_result = results[0]
            content_preview = first_result.get("content", "")[:100] if first_result.get("content") else "No content"
            logger.info(f"First result preview: {content_preview}...")
        
        # Skip similarity filtering - just use all results
        filtered_results = results
        
        logger.info(f"Retrieved {len(filtered_results)} documents for query")
        return filtered_results
    
    def get_relevant_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
    ) -> str:
        """
        Get relevant context for a query as a string.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_dict: Dictionary of metadata filters
            
        Returns:
            str: Concatenated content of retrieved chunks
        """
        # Retrieve relevant chunks
        results = self.retrieve(query=query, top_k=top_k, filter_dict=filter_dict)
        
        # Debug log the raw results
        logger.info(f"Raw results count: {len(results)}")
        for i, result in enumerate(results):
            chunk_id = result.get("chunk_id", "unknown")
            similarity = result.get("similarity", 0)
            content_preview = result.get("content", "")[:50] if result.get("content") else "No content"
            logger.info(f"Result {i+1}: chunk_id={chunk_id}, similarity={similarity:.4f}, preview={content_preview}...")
        
        # TEMPORARY FIX: Even if results are empty, create a fallback response
        if not results:
            # Force a search with increased top_k and no filtering
            logger.warning("No results found. Attempting broader search...")
            # Search vector store directly with increased top_k
            query_embedding = self.embedding_service.embed_query(query)
            direct_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=10,  # Increased from default
                filter_dict=None,  # No filtering
            )
            
            if direct_results:
                logger.info(f"Direct search found {len(direct_results)} results")
                results = direct_results
            else:
                logger.warning("No relevant context found for query even with broad search")
                return ""
            
        # Format chunks with metadata
        formatted_chunks = []
        
        for i, result in enumerate(results):
            content = result.get("content", "No content available")
            metadata = result.get("metadata", {})
            similarity = result.get("similarity", 0)
            
            # Include relevant metadata in context
            source = metadata.get("source", "Unknown")
            if "page_count" in metadata and "chunk_index" in metadata:
                page_info = f"Page {metadata.get('chunk_index', 0) + 1} of {metadata.get('page_count', 1)}"
            else:
                page_info = ""
                
            # Format chunk with metadata
            formatted_chunk = (
                f"[Document {i+1}] {source} {page_info}\n"
                f"{content}\n"
                f"Relevance: {similarity:.2f}\n"
            )
            
            formatted_chunks.append(formatted_chunk)
            
        # Combine chunks into a single context string
        context = "\n\n".join(formatted_chunks)
        logger.info(f"Created context with {len(formatted_chunks)} chunks, total length: {len(context)} characters")
        
        return context 