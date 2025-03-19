"""
Ragie Integration Module
------------------------
This module provides integration with Ragie's managed RAG service.
It implements a client for Ragie's API and adapters to use it with our system.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union, BinaryIO
from pathlib import Path

try:
    from ragie import Ragie as RagieSDK
    from ragie.models import SDKError
    RAGIE_AVAILABLE = True
except ImportError:
    RAGIE_AVAILABLE = False
    RagieSDK = object  # Placeholder for type checking
    class SDKError(Exception):
        pass

logger = logging.getLogger(__name__)

class RagieClient:
    """Client for interacting with the Ragie API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Ragie client.
        
        Args:
            api_key: The Ragie API key. If not provided, will look for RAGIE_API_KEY in environment.
        
        Raises:
            ImportError: If the ragie package is not installed
            ValueError: If no API key is provided or found in environment
        """
        if not RAGIE_AVAILABLE:
            raise ImportError(
                "The ragie package is not installed. Please install it with 'pip install ragie'."
            )
        
        self.api_key = api_key or os.environ.get("RAGIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Ragie API key provided. Please provide an API key or set the RAGIE_API_KEY environment variable."
            )
        
        # Initialize the Ragie SDK client
        self.client = RagieSDK(auth=self.api_key)
        logger.info("Ragie client initialized")
    
    def upload_document(self, 
                        file_path: Union[str, Path], 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Upload a document to Ragie.
        
        Args:
            file_path: Path to the file to upload
            metadata: Optional metadata to attach to the document
            
        Returns:
            Dict containing document information including ID and status
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Uploading document: {file_path}")
        
        try:
            with open(file_path, "rb") as file:
                request = {
                    "file": {
                        "file_name": file_path.name,
                        "content": file
                    }
                }
                
                if metadata:
                    request["metadata"] = metadata
                
                response = self.client.documents.create(request=request)
                
                return {
                    "id": response.id,
                    "status": response.status,
                    "metadata": metadata or {}
                }
        except SDKError as e:
            logger.error(f"Error uploading document: {e}")
            raise

    def upload_document_from_bytes(self, 
                                  file_content: bytes,
                                  file_name: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Upload a document to Ragie from bytes content.
        
        Args:
            file_content: Bytes content of the file
            file_name: Name to give the file
            metadata: Optional metadata to attach to the document
            
        Returns:
            Dict containing document information including ID and status
        """
        logger.info(f"Uploading document from bytes: {file_name}")
        
        try:
            request = {
                "file": {
                    "file_name": file_name,
                    "content": file_content
                }
            }
            
            if metadata:
                request["metadata"] = metadata
            
            response = self.client.documents.create(request=request)
            
            return {
                "id": response.id,
                "status": response.status,
                "metadata": metadata or {}
            }
        except SDKError as e:
            logger.error(f"Error uploading document: {e}")
            raise

    def get_document_status(self, document_id: str) -> str:
        """
        Get the processing status of a document.
        
        Args:
            document_id: The ID of the document
            
        Returns:
            The status of the document (e.g., "pending", "ready", "failed")
        """
        try:
            response = self.client.documents.get(document_id)
            return response.status
        except SDKError as e:
            logger.error(f"Error getting document status: {e}")
            raise

    def wait_for_document_ready(self, document_id: str, timeout: int = 300, interval: int = 5) -> str:
        """
        Wait for a document to be fully processed and ready.
        
        Args:
            document_id: The ID of the document
            timeout: Maximum seconds to wait
            interval: Polling interval in seconds
            
        Returns:
            The final status of the document
            
        Raises:
            TimeoutError: If the document does not become ready within the timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_document_status(document_id)
            if status == "ready":
                return status
            if status == "failed":
                raise ValueError(f"Document processing failed: {document_id}")
            logger.info(f"Document {document_id} status: {status}. Waiting...")
            time.sleep(interval)
        
        raise TimeoutError(f"Timed out waiting for document {document_id} to become ready")

    def retrieve(self, 
                query: str, 
                document_ids: Optional[List[str]] = None,
                filter_metadata: Optional[Dict[str, Any]] = None,
                rerank: bool = True,
                top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The query text
            document_ids: Optional list of document IDs to search within
            filter_metadata: Optional metadata filter
            rerank: Whether to rerank results for improved relevance
            top_k: Number of results to return
            
        Returns:
            Dict containing retrieval results
        """
        try:
            # Build payload according to Ragie API reference
            payload = {
                "query": query  # Required parameter
            }
            
            # Add optional parameters
            if document_ids:
                logger.info(f"Filtering by document_ids: {document_ids}")
                payload["document_ids"] = document_ids
                
            if filter_metadata:
                # Convert filter_metadata to proper Ragie format if needed
                if isinstance(filter_metadata, dict):
                    # If it's already a dict, keep it as is
                    filter_json = filter_metadata
                    logger.info(f"Using filter_metadata: {filter_json}")
                else:
                    # Otherwise try to parse it as JSON
                    import json
                    try:
                        filter_json = json.loads(filter_metadata)
                        logger.info(f"Parsed filter_metadata: {filter_json}")
                    except Exception as e:
                        logger.warning(f"Failed to parse filter_metadata as JSON: {e}")
                        filter_json = filter_metadata
                
                payload["filter"] = filter_json
                
            # Always set rerank explicitly since the default is False in the API
            payload["rerank"] = rerank
                
            if top_k is not None:
                # Parameter is called top_k in the API reference
                payload["top_k"] = top_k
            else:
                # Set a reasonable default if not specified
                payload["top_k"] = 5
            
            # Add recency_bias parameter to favor more recent documents
            payload["recency_bias"] = True
            
            logger.info(f"Retrieving results for query: '{query}' with payload: {payload}")
            
            # Pass the payload as a keyword argument named 'request'
            # This is the key fix - Ragie SDK expects a keyword argument named 'request'
            response = self.client.retrievals.retrieve(request=payload)
            
            # Log response type for debugging
            logger.info(f"Retrieval response type: {type(response)}")
            
            # Log available attributes on the response
            if hasattr(response, "__dict__"):
                logger.info(f"Response attributes: {list(response.__dict__.keys())}")
            
            # Try to extract chunks from different possible locations
            chunks = []
            
            # Based on the logs, 'scored_chunks' appears to be the attribute to check first
            if hasattr(response, "scored_chunks"):
                logger.info(f"Found scored_chunks in response")
                chunks = response.scored_chunks
            # Check for chunks attribute (most likely based on API docs)
            elif hasattr(response, "chunks"):
                logger.info(f"Found chunks in response.chunks")
                chunks = response.chunks
            # Alternative attribute names
            elif hasattr(response, "matches"):
                logger.info(f"Found matches in response.matches")
                chunks = response.matches
            elif hasattr(response, "results"):
                logger.info(f"Found results in response.results")
                chunks = response.results
            # Check if result is nested one level deeper
            elif hasattr(response, "result"):
                result = response.result
                if hasattr(result, "chunks"):
                    logger.info(f"Found chunks in response.result.chunks")
                    chunks = result.chunks
                elif hasattr(result, "matches"):
                    logger.info(f"Found matches in response.result.matches")
                    chunks = result.matches
                elif hasattr(result, "scored_chunks"):
                    logger.info(f"Found scored_chunks in response.result.scored_chunks")
                    chunks = result.scored_chunks
            
            # If still no chunks found, try to iterate response directly
            if not chunks and hasattr(response, "__iter__"):
                logger.info("Attempting to iterate response directly for chunks")
                try:
                    chunks = list(response)
                    logger.info(f"Found {len(chunks)} chunks by direct iteration")
                    # If we're here, let's log a sample chunk to understand its structure
                    if len(chunks) > 0:
                        sample_chunk = chunks[0]
                        logger.info(f"Sample chunk type: {type(sample_chunk)}")
                        if hasattr(sample_chunk, "__dict__"):
                            logger.info(f"Sample chunk attributes: {list(sample_chunk.__dict__.keys())}")
                except Exception as e:
                    logger.error(f"Failed to iterate response: {e}")
            
            if not chunks:
                logger.warning(f"No chunks found in response for query: '{query}'")
                return {"chunks": []}
            
            # Log the number of chunks found
            logger.info(f"Found {len(chunks)} raw chunks from retrieval")
            
            # Extract chunk data with flexible attribute access
            result_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    # Log chunk data for debugging
                    logger.debug(f"Processing chunk {i}")
                    if hasattr(chunk, "__dict__"):
                        logger.debug(f"Chunk {i} attributes: {list(chunk.__dict__.keys())}")
                    
                    # Try multiple attribute names for text content
                    text = ""
                    for attr_name in ["text", "content", "chunk", "document_content"]:
                        if hasattr(chunk, attr_name) and getattr(chunk, attr_name):
                            text = getattr(chunk, attr_name)
                            logger.debug(f"Found text in .{attr_name}")
                            break
                    
                    # Try multiple attribute names for score
                    score = 0.0
                    for attr_name in ["score", "similarity", "relevance", "confidence"]:
                        if hasattr(chunk, attr_name) and getattr(chunk, attr_name) is not None:
                            try:
                                score = float(getattr(chunk, attr_name))
                                logger.debug(f"Found score in .{attr_name}: {score}")
                                break
                            except (ValueError, TypeError):
                                pass
                    
                    # Try multiple attribute names for metadata
                    metadata = {}
                    for attr_name in ["metadata", "meta", "properties"]:
                        if hasattr(chunk, attr_name) and getattr(chunk, attr_name):
                            metadata = getattr(chunk, attr_name)
                            logger.debug(f"Found metadata in .{attr_name}")
                            break
                    
                    # Try multiple attribute names for document ID
                    document_id = "unknown"
                    for attr_name in ["document_id", "id", "doc_id", "documentId"]:
                        if hasattr(chunk, attr_name) and getattr(chunk, attr_name):
                            document_id = getattr(chunk, attr_name)
                            logger.debug(f"Found document_id in .{attr_name}: {document_id}")
                            break
                    
                    # Build a complete chunk data object
                    chunk_data = {
                        "text": text,
                        "score": score,
                        "metadata": metadata,
                        "document_id": document_id
                    }
                    
                    # Make sure we only add chunks that actually have text content
                    if chunk_data["text"] and chunk_data["text"].strip():
                        result_chunks.append(chunk_data)
                        logger.debug(f"Added chunk {i} with score {score}")
                    else:
                        logger.debug(f"Skipped chunk {i} due to empty text")
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
            
            logger.info(f"Successfully processed {len(result_chunks)} chunks from retrieval")
            return {"chunks": result_chunks}
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            raise

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from Ragie.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.documents.delete(document_id)
            return True
        except SDKError as e:
            logger.error(f"Error deleting document: {e}")
            raise

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the Ragie account.
        
        Returns:
            List of document information dictionaries
        """
        try:
            # Get the documents without specifying page_size (not supported)
            logger.info("Requesting documents from Ragie API")
            response = self.client.documents.list()
            documents = []
            
            # Log response type for debugging
            logger.info(f"Ragie documents.list() response type: {type(response)}")
            
            # Debug the response structure
            if hasattr(response, "__dict__"):
                logger.info(f"Response attributes: {list(response.__dict__.keys())}")
            
            # Try to extract documents from the response based on its structure
            if hasattr(response, "result") and response.result:
                logger.info(f"Found documents in response.result")
                result_data = response.result
                
                # Check if result contains document items
                if hasattr(result_data, "items") and result_data.items:
                    for doc in result_data.items:
                        try:
                            doc_data = {
                                "id": getattr(doc, "id", "unknown"),
                                "status": getattr(doc, "status", "unknown"),
                                "metadata": getattr(doc, "metadata", {})
                            }
                            documents.append(doc_data)
                        except Exception as e:
                            logger.error(f"Error processing document item: {e}")
            
            # If we haven't found documents yet, check for other common structures
            if not documents:
                # Try different attribute names or direct iteration
                for attr_name in ["items", "documents", "docs", "data"]:
                    if hasattr(response, attr_name):
                        attr_value = getattr(response, attr_name)
                        if attr_value:
                            logger.info(f"Found documents in response.{attr_name}")
                            try:
                                for doc in attr_value:
                                    doc_data = {
                                        "id": getattr(doc, "id", "unknown"),
                                        "status": getattr(doc, "status", "unknown"), 
                                        "metadata": getattr(doc, "metadata", {})
                                    }
                                    documents.append(doc_data)
                            except Exception as e:
                                logger.error(f"Error extracting from {attr_name}: {e}")
                            break
            
            # If still no documents, try direct iteration if possible
            if not documents and hasattr(response, "__iter__"):
                try:
                    logger.info("Attempting direct iteration of response")
                    for doc in response:
                        try:
                            doc_data = {
                                "id": getattr(doc, "id", "unknown"),
                                "status": getattr(doc, "status", "unknown"),
                                "metadata": getattr(doc, "metadata", {})
                            }
                            documents.append(doc_data)
                        except Exception as e:
                            logger.error(f"Error in direct iteration: {e}")
                except Exception as e:
                    logger.error(f"Failed to iterate response: {e}")
            
            # Handle pagination with a limit to avoid infinite loops
            max_pages = 3
            current_page = 1
            
            if hasattr(response, "next") and callable(response.next):
                try:
                    next_page = response.next()
                    while next_page and current_page < max_pages:
                        logger.info(f"Processing page {current_page + 1}")
                        current_page += 1
                        
                        # Extract documents similar to the first page
                        if hasattr(next_page, "result") and next_page.result:
                            result_data = next_page.result
                            if hasattr(result_data, "items"):
                                for doc in result_data.items:
                                    try:
                                        doc_data = {
                                            "id": getattr(doc, "id", "unknown"),
                                            "status": getattr(doc, "status", "unknown"),
                                            "metadata": getattr(doc, "metadata", {})
                                        }
                                        documents.append(doc_data)
                                    except Exception as e:
                                        logger.error(f"Error processing item from page {current_page}: {e}")
                        
                        # Try to get the next page
                        try:
                            next_page = next_page.next()
                        except Exception as e:
                            logger.warning(f"Error getting next page: {e}")
                            break
                except Exception as e:
                    logger.error(f"Error in pagination: {e}")
            
            # If no documents found and no errors, return empty list
            logger.info(f"Retrieved {len(documents)} documents from Ragie")
            return documents
        except SDKError as e:
            logger.error(f"Error listing documents: {e}")
            raise 