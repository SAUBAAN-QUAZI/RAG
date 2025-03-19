"""
Ragie-based RAG Agent Module
--------------------------
This module implements a RAG agent that uses Ragie for document processing and retrieval.
"""

import time
import logging
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json

from pydantic import BaseModel, Field

from ..config import (
    OPENAI_API_KEY, USE_RAGIE, MAX_RESPONSE_TOKENS, TOP_K_RESULTS
)
from ..integrations.ragie import RagieClient

# Set up logger
logger = logging.getLogger(__name__)

class RagieRAGAgent:
    """
    RAG Agent that uses Ragie for document processing and retrieval
    """
    
    def __init__(self):
        """
        Initialize the Ragie-based RAG agent
        
        Raises:
            ValueError: If Ragie is not enabled in config or if initialization fails
        """
        if not USE_RAGIE:
            raise ValueError("Ragie is not enabled in configuration. Set USE_RAGIE=True to use this agent.")
        
        try:
            # Initialize the OpenAI client for generating responses
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize the Ragie client
            self.ragie_client = RagieClient()
            
            # For timing different phases
            self.start_time = None
            self.timings = {}
            
            logger.info("RagieRAGAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RagieRAGAgent: {e}")
            raise
    
    def add_document(self, 
                     file_path: Union[str, Path], 
                     document_id: Optional[str] = None,
                     document_name: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a document to the Ragie system
        
        Args:
            file_path: Path to the document file
            document_id: Optional document ID (if not provided, Ragie will generate one)
            document_name: Optional document name for display
            metadata: Optional metadata to attach to the document
            
        Returns:
            Dictionary with document information including id and status
        """
        file_path = Path(file_path)
        
        # Create metadata dict if not provided
        if metadata is None:
            metadata = {}
        
        # Add document_name to metadata if provided
        if document_name:
            metadata["document_name"] = document_name
        else:
            metadata["document_name"] = file_path.name
        
        logger.info(f"Adding document: {file_path}")
        
        try:
            # Upload document to Ragie
            result = self.ragie_client.upload_document(file_path, metadata=metadata)
            
            # Wait for document to be fully processed
            document_id = result["id"]
            logger.info(f"Document uploaded with ID: {document_id}, waiting for processing...")
            
            self.ragie_client.wait_for_document_ready(document_id)
            logger.info(f"Document {document_id} is ready for retrieval")
            
            return result
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query before sending to Ragie retrieval
        
        Args:
            query: The raw user query
            
        Returns:
            Preprocessed query
        """
        # For now, just return the original query
        # This is a placeholder for future query preprocessing logic
        return query
    
    def query(self, 
              query: str, 
              document_ids: Optional[List[str]] = None,
              filter_metadata: Optional[Dict[str, Any]] = None,
              top_k: Optional[int] = None,
              show_timings: bool = False) -> Dict[str, Any]:
        """
        Process a query using the Ragie RAG pipeline
        
        Args:
            query: The query text
            document_ids: Optional list of specific document IDs to search within
            filter_metadata: Optional metadata filter to narrow search
            top_k: Number of chunks to retrieve (default: from config)
            show_timings: Whether to include timing information in the response
            
        Returns:
            Dictionary with query response
        """
        # Start timing
        self.start_time = time.time()
        self.timings = {}
        
        try:
            # Ensure we have documents before trying to query
            all_docs = self.list_documents()
            if not all_docs:
                logger.warning("No documents found in the system")
                return {
                    "query": query,
                    "response": "No documents have been uploaded to the system yet. Please upload documents before asking questions.",
                    "chunks": [],
                    "document_ids": document_ids or []
                }
                
            # Preprocess the query
            preprocessed_query = self._preprocess_query(query)
            self.timings["preprocess"] = time.time() - self.start_time
            
            # Get relevant chunks from Ragie
            retrieval_start = time.time()
            
            if top_k is None:
                top_k = TOP_K_RESULTS
                
            # Try to retrieve relevant chunks
            retrieval_results = self.ragie_client.retrieve(
                query=preprocessed_query,
                document_ids=document_ids,
                filter_metadata=filter_metadata,
                top_k=top_k,
                rerank=True
            )
            self.timings["retrieval"] = time.time() - retrieval_start
            
            # Generate relevant context from the retrieved chunks
            context_start = time.time()
            context = self._get_relevant_context(retrieval_results["chunks"], query)
            self.timings["context"] = time.time() - context_start
            
            # Check if we have any useful context
            has_relevant_info = bool(retrieval_results["chunks"]) and not context.startswith("No relevant information")
            
            # Generate response using LLM
            response_start = time.time()
            if has_relevant_info:
                response = self._generate_response(query, context)
            else:
                # Provide a helpful response when no relevant info is found
                response = f"I don't have enough information in the provided context to explain {query}. Please provide additional details or context, and I would be happy to help with your query."
            self.timings["response"] = time.time() - response_start
            
            # Total time
            self.timings["total"] = time.time() - self.start_time
            
            # Prepare the result
            result = {
                "query": query,
                "response": response,
                "chunks": retrieval_results["chunks"],
                "document_ids": document_ids or []
            }
            
            if show_timings:
                result["timings"] = self.timings
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def _get_relevant_context(self, chunks: List[Dict[str, Any]], query: str) -> str:
        """
        Format retrieved chunks into a context string for the LLM
        
        Args:
            chunks: List of chunk dictionaries from retrieval
            query: The original query
            
        Returns:
            Formatted context string
        """
        if not chunks:
            logger.warning(f"No relevant chunks found for query: '{query}'")
            return "No relevant information was found in the documents. The system has no knowledge about this topic based on the uploaded documents."
        
        # Format the chunks into a context string
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            score = chunk.get("score", 0)
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            doc_id = chunk.get("document_id", "unknown")
            
            # Skip empty chunks
            if not text.strip():
                continue
                
            # Get document name from metadata if available
            doc_name = metadata.get("document_name", f"Document {doc_id}")
            
            # Format chunk with source information
            context_parts.append(
                f"[Source {i+1}: {doc_name}, Relevance: {score:.2f}]\n{text}\n"
            )
        
        # If all chunks were empty, return no information message
        if not context_parts:
            logger.warning("All retrieved chunks were empty")
            return "The system found document matches, but they don't contain relevant text content for this query."
            
        return "\n\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using the LLM based on the query and context
        
        Args:
            query: The user query
            context: The retrieved context
            
        Returns:
            Generated response text
        """
        # Prepare the prompt
        prompt = f"""
        You are an AI assistant tasked with answering questions based on the provided context.
        
        Context:
        {context}
        
        Question: {query}
        
        Provide a comprehensive answer based on the context provided. If the context doesn't contain 
        the information needed to answer the question, state that you don't have enough information.
        Do not make up information that is not supported by the context.
        
        Answer:
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",  # Can be configured based on needs
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_RESPONSE_TOKENS,
                temperature=0.2,  # Lower temperature for more factual responses
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the Ragie system
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if successful
        """
        return self.ragie_client.delete_document(document_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the Ragie system
        
        Returns:
            List of document dictionaries
        """
        return self.ragie_client.get_all_documents() 