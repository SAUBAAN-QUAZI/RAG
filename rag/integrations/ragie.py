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
    
    def __init__(self, api_key: Optional[str] = None, default_partition: Optional[str] = None):
        """
        Initialize the Ragie client.
        
        Args:
            api_key: The Ragie API key. If not provided, will look for RAGIE_API_KEY in environment.
            default_partition: Default partition to use for documents if not specified in methods.
        
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
        
        # Store default partition
        self.default_partition = default_partition
        
        # Initialize the Ragie SDK client
        self.client = RagieSDK(auth=self.api_key)
        logger.info("Ragie client initialized")
    
    def upload_document(self, 
                       file_path: Union[str, Path], 
                       metadata: Optional[Dict[str, Any]] = None,
                       mode: str = "fast",
                       external_id: Optional[str] = None,
                       name: Optional[str] = None,
                       partition: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a document to Ragie.
        
        Args:
            file_path: Path to the file to upload
            metadata: Optional metadata to attach to the document
            mode: Document processing mode ('fast' or 'hi_res')
            external_id: Optional external identifier for the document
            name: Optional name for the document (defaults to filename)
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            Dict containing document information including ID and status
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Uploading document: {file_path} with mode: {mode}")
        
        try:
            with open(file_path, "rb") as file:
                # Build the request based on Ragie API spec
                request = {
                    "file": {
                        "file_name": file_path.name,
                        "content": file
                    },
                    "mode": mode
                }
                
                # Add optional parameters
                if metadata:
                    request["metadata"] = metadata
                
                if external_id:
                    request["external_id"] = external_id
                
                if name:
                    request["name"] = name
                
                # Use provided partition or default
                if partition or self.default_partition:
                    request["partition"] = partition or self.default_partition
                
                # Call the Ragie API
                response = self.client.documents.create(request=request)
                
                return {
                    "id": response.id,
                    "status": response.status,
                    "metadata": metadata or {},
                    "name": name or file_path.name,
                    "external_id": external_id
                }
        except SDKError as e:
            logger.error(f"Error uploading document: {e}")
            raise

    def upload_document_from_bytes(self, 
                                  file_content: bytes,
                                  file_name: str,
                                  metadata: Optional[Dict[str, Any]] = None,
                                  mode: str = "fast",
                                  external_id: Optional[str] = None,
                                  name: Optional[str] = None,
                                  partition: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a document to Ragie from bytes content.
        
        Args:
            file_content: Bytes content of the file
            file_name: Name to give the file
            metadata: Optional metadata to attach to the document
            mode: Document processing mode ('fast' or 'hi_res')
            external_id: Optional external identifier for the document
            name: Optional name for the document
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            Dict containing document information including ID and status
        """
        logger.info(f"Uploading document from bytes: {file_name} with mode: {mode}")
        
        try:
            # Build the request based on Ragie API spec
            request = {
                "file": {
                    "file_name": file_name,
                    "content": file_content
                },
                "mode": mode
            }
            
            # Add optional parameters
            if metadata:
                request["metadata"] = metadata
            
            if external_id:
                request["external_id"] = external_id
            
            if name:
                request["name"] = name
            
            # Use provided partition or default
            if partition or self.default_partition:
                request["partition"] = partition or self.default_partition
            
            # Call the Ragie API
            response = self.client.documents.create(request=request)
            
            return {
                "id": response.id,
                "status": response.status,
                "metadata": metadata or {},
                "name": name or file_name,
                "external_id": external_id
            }
        except SDKError as e:
            logger.error(f"Error uploading document: {e}")
            raise

    def upload_document_raw(self,
                           data: str,
                           name: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           external_id: Optional[str] = None,
                           partition: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a document to Ragie as raw text.
        
        Args:
            data: The raw text content
            name: Name for the document (defaults to timestamp)
            metadata: Optional metadata to attach to the document
            external_id: Optional external identifier for the document
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            Dict containing document information including ID and status
        """
        logger.info(f"Uploading raw text document: {name or 'unnamed'}")
        
        try:
            # Build the request based on Ragie API spec
            request = {
                "data": data
            }
            
            # Add optional parameters
            if metadata:
                request["metadata"] = metadata
            
            if external_id:
                request["external_id"] = external_id
            
            if name:
                request["name"] = name
            
            # Use provided partition or default
            if partition or self.default_partition:
                request["partition"] = partition or self.default_partition
            
            # Call the Ragie API
            response = self.client.documents.create_raw(request=request)
            
            return {
                "id": response.id,
                "status": response.status,
                "metadata": metadata or {},
                "name": name,
                "external_id": external_id
            }
        except SDKError as e:
            logger.error(f"Error uploading raw document: {e}")
            raise

    def upload_document_from_url(self,
                                url: str,
                                name: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None,
                                mode: str = "fast",
                                external_id: Optional[str] = None,
                                partition: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a document to Ragie from a publicly accessible URL.
        
        Args:
            url: The URL of the document to download
            name: Optional name for the document
            metadata: Optional metadata to attach to the document
            mode: Document processing mode ('fast' or 'hi_res')
            external_id: Optional external identifier for the document
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            Dict containing document information including ID and status
        """
        logger.info(f"Uploading document from URL: {url} with mode: {mode}")
        
        try:
            # Build the request based on Ragie API spec
            request = {
                "url": url,
                "mode": mode
            }
            
            # Add optional parameters
            if metadata:
                request["metadata"] = metadata
            
            if external_id:
                request["external_id"] = external_id
            
            if name:
                request["name"] = name
            
            # Use provided partition or default
            if partition or self.default_partition:
                request["partition"] = partition or self.default_partition
            
            # Call the Ragie API
            response = self.client.documents.create_from_url(request=request)
            
            return {
                "id": response.id,
                "status": response.status,
                "metadata": metadata or {},
                "name": name,
                "external_id": external_id
            }
        except SDKError as e:
            logger.error(f"Error uploading document from URL: {e}")
            raise

    def get_document_status(self, document_id: str, partition: Optional[str] = None) -> str:
        """
        Get the processing status of a document.
        
        Args:
            document_id: The ID of the document
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            The status of the document (e.g., "pending", "ready", "failed")
        """
        try:
            # Get the partition to use but don't pass it as headers
            partition_val = partition or self.default_partition
            
            # SDK v1.5.0 document.get() method doesn't take document_id as a positional argument
            # Let's modify the calling approach
            logger.info(f"Getting status for document {document_id}")
            
            # In SDK v1.5.0, we may need to use document_id as a parameter name
            # or as part of a request dict
            try:
                # First try with named parameter
                response = self.client.documents.get(document_id=document_id)
            except TypeError:
                try:
                    # Try with a request dict
                    response = self.client.documents.get(request={"document_id": document_id})
                except TypeError:
                    # Try using the SDK's GetDocumentOp instead (if available)
                    from ragie.models import GetDocumentOp
                    op = GetDocumentOp(document_id=document_id)
                    response = self.client.client.execute(op)
            
            # Extract status based on response structure
            if hasattr(response, "status"):
                return response.status
            elif hasattr(response, "document") and hasattr(response.document, "status"):
                return response.document.status
            else:
                logger.warning(f"Could not extract status from response")
                return "unknown"
        except SDKError as e:
            logger.error(f"Error getting document status: {e}")
            raise

    def wait_for_document_ready(self, 
                              document_id: str, 
                              accept_indexed: bool = True,
                              timeout: int = 300, 
                              interval: int = 5,
                              partition: Optional[str] = None) -> str:
        """
        Wait for a document to be fully processed and ready.
        
        Args:
            document_id: The ID of the document
            accept_indexed: Whether to accept 'indexed' state as ready
            timeout: Maximum seconds to wait
            interval: Polling interval in seconds
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            The final status of the document
            
        Raises:
            TimeoutError: If the document does not become ready within the timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_document_status(document_id, partition=partition)
            
            # Check if status is ready or at least indexed if accept_indexed is True
            if status == "ready" or (accept_indexed and status == "indexed"):
                return status
                
            # Handle failed status
            if status == "failed":
                raise ValueError(f"Document processing failed: {document_id}")
                
            # List of states in processing order
            status_sequence = [
                "pending", 
                "partitioning", 
                "partitioned", 
                "refined", 
                "chunked", 
                "indexed", 
                "summary_indexed", 
                "keyword_indexed", 
                "ready"
            ]
            
            # Log current status with context about progression
            try:
                current_idx = status_sequence.index(status)
                total_steps = len(status_sequence)
                logger.info(f"Document {document_id} status: {status} (Step {current_idx+1}/{total_steps}). Waiting...")
            except ValueError:
                # Status not in our known sequence
                logger.info(f"Document {document_id} status: {status} (unknown step). Waiting...")
                
            time.sleep(interval)
        
        raise TimeoutError(f"Timed out waiting for document {document_id} to become ready")

    def retrieve(self, 
                  query: str, 
                  top_k: int = 5, 
                  filter_metadata: Optional[Dict[str, Any]] = None,
                  document_ids: Optional[List[str]] = None,
                  rerank: bool = True,
                  partition: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve document chunks relevant to a query.
        
        Args:
            query: The search query
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filter
            document_ids: Optional list of document IDs to search within
            rerank: Whether to rerank results
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            Dict containing retrieved chunks
        """
        try:
            # Prepare the request payload
            payload = {
                "query": query,
                "top_k": top_k,
                "rerank": rerank
            }
            
            # Add optional filter parameters
            if filter_metadata:
                payload["filter"] = filter_metadata
                
            if document_ids:
                payload["document_ids"] = document_ids
            
            # Get partition value but don't pass it as headers
            partition_val = partition or self.default_partition
            
            logger.info(f"Retrieving results for query: '{query}' with payload: {payload}")
            
            # Pass the payload as a keyword argument named 'request'
            # SDK v1.5.0 doesn't support headers parameter
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

    def delete_document(self, document_id: str, partition: Optional[str] = None) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: The ID of the document to delete
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            True if deletion was successful
        """
        try:
            # Get the partition to use (but don't pass it as header param)
            partition_val = partition or self.default_partition
            
            # Call the Ragie API without headers parameter
            logger.info(f"Deleting document {document_id}")
            response = self.client.documents.delete(document_id)
            
            # Check if deletion was successful based on response structure
            if hasattr(response, "status") and response.status in ["success", "deleted"]:
                logger.info(f"Document {document_id} deleted successfully")
                return True
            elif hasattr(response, "success") and response.success:
                logger.info(f"Document {document_id} deleted successfully")
                return True
            else:
                # If we can't determine success from response, assume it worked if no error was raised
                logger.info(f"Document {document_id} assumed deleted (no error)")
                return True
                
        except SDKError as e:
            logger.error(f"Error deleting document: {e}")
            raise

    def get_all_documents(self, partition: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all documents in the Ragie account.
        
        Args:
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            List of document information dictionaries
        """
        try:
            # Set the partition to use
            partition_val = partition or self.default_partition
            
            # Get the documents with pagination parameters
            logger.info("Requesting documents from Ragie API")
            
            # In version 1.5.0, the SDK doesn't accept page_size parameter
            # We'll call list() without parameters
            params = {}
            
            # Add partition if available
            if partition_val:
                params["partition"] = partition_val
            
            # Call the list method
            response = self.client.documents.list(**params)
            
            # Log response structure for debugging
            logger.info(f"Document list response type: {type(response)}")
            if hasattr(response, "__dict__"):
                logger.info(f"Response attributes: {list(response.__dict__.keys())}")
            elif hasattr(response, "__dir__"):
                logger.info(f"Response attributes: {dir(response)}")
            
            documents = []
            
            # Process response to extract documents based on SDK's response structure
            if hasattr(response, "documents"):
                # Direct documents list
                logger.info(f"Found {len(response.documents)} documents in response.documents")
                for doc in response.documents:
                    try:
                        # Log document structure
                        if hasattr(doc, "__dict__"):
                            logger.info(f"Document attributes: {list(doc.__dict__.keys())}")
                        elif hasattr(doc, "__dir__"):
                            logger.info(f"Document attributes: {dir(doc)}")
                        
                        # Extract document data with a more flexible approach
                        doc_data = {}
                        
                        # Try different attribute names for ID
                        for id_attr in ["id", "document_id", "_id", "uuid"]:
                            if hasattr(doc, id_attr):
                                doc_data["id"] = getattr(doc, id_attr)
                                break
                        
                        # Try different attribute names for status
                        for status_attr in ["status", "state", "processing_status"]:
                            if hasattr(doc, status_attr):
                                doc_data["status"] = getattr(doc, status_attr)
                                break
                        
                        # Try different attribute names for metadata
                        for meta_attr in ["metadata", "meta", "properties"]:
                            if hasattr(doc, meta_attr):
                                doc_data["metadata"] = getattr(doc, meta_attr) or {}
                                break
                        
                        # Try different attribute names for name/title
                        for name_attr in ["name", "title", "document_name", "filename"]:
                            if hasattr(doc, name_attr):
                                doc_data["name"] = getattr(doc, name_attr)
                                break
                        
                        # Try different attribute names for creation time
                        for time_attr in ["created_at", "creation_time", "created"]:
                            if hasattr(doc, time_attr):
                                doc_data["created_at"] = getattr(doc, time_attr)
                                break
                                
                        # Ensure we have at least these keys with defaults
                        if "id" not in doc_data:
                            doc_data["id"] = "unknown"
                        if "status" not in doc_data:
                            doc_data["status"] = "unknown"
                        if "metadata" not in doc_data:
                            doc_data["metadata"] = {}
                            
                        documents.append(doc_data)
                    except Exception as e:
                        logger.error(f"Error processing document: {e}")
            
            # Try direct iteration if no documents attribute
            elif hasattr(response, "__iter__"):
                try:
                    docs_list = list(response)
                    logger.info(f"Found {len(docs_list)} documents by direct iteration")
                    
                    # If we have items, log the first one to understand structure
                    if docs_list:
                        sample_doc = docs_list[0]
                        if hasattr(sample_doc, "__dict__"):
                            logger.info(f"Sample document attributes: {list(sample_doc.__dict__.keys())}")
                        elif hasattr(sample_doc, "__dir__"):
                            logger.info(f"Sample document attributes: {dir(sample_doc)}")
                    
                    for doc in docs_list:
                        try:
                            # Extract document data with a more flexible approach
                            doc_data = {}
                            
                            # Try different attribute names for ID
                            for id_attr in ["id", "document_id", "_id", "uuid"]:
                                if hasattr(doc, id_attr):
                                    doc_data["id"] = getattr(doc, id_attr)
                                    break
                            
                            # Try different attribute names for status
                            for status_attr in ["status", "state", "processing_status"]:
                                if hasattr(doc, status_attr):
                                    doc_data["status"] = getattr(doc, status_attr)
                                    break
                            
                            # Try different attribute names for metadata
                            for meta_attr in ["metadata", "meta", "properties"]:
                                if hasattr(doc, meta_attr):
                                    doc_data["metadata"] = getattr(doc, meta_attr) or {}
                                    break
                            
                            # Try different attribute names for name/title
                            for name_attr in ["name", "title", "document_name", "filename"]:
                                if hasattr(doc, name_attr):
                                    doc_data["name"] = getattr(doc, name_attr)
                                    break
                            
                            # Ensure we have at least these keys with defaults
                            if "id" not in doc_data:
                                doc_data["id"] = "unknown"
                            if "status" not in doc_data:
                                doc_data["status"] = "unknown"
                            if "metadata" not in doc_data:
                                doc_data["metadata"] = {}
                                
                            documents.append(doc_data)
                        except Exception as e:
                            logger.error(f"Error processing document in iteration: {e}")
                except Exception as e:
                    logger.error(f"Error iterating response: {e}")
            
            # Handle pagination with cursor if available
            cursor = getattr(response, "cursor", None)
            
            # Limit pagination to avoid infinite loops or excessive requests
            max_pages = 5
            current_page = 1
            
            while cursor and current_page < max_pages:
                try:
                    logger.info(f"Fetching next page of documents with cursor")
                    
                    # Use cursor for pagination
                    params["cursor"] = cursor
                    next_response = self.client.documents.list(**params)
                    
                    # Process documents from the next page
                    if hasattr(next_response, "documents"):
                        for doc in next_response.documents:
                            try:
                                # Extract document data with a more flexible approach
                                doc_data = {}
                                
                                # Try different attribute names for ID
                                for id_attr in ["id", "document_id", "_id", "uuid"]:
                                    if hasattr(doc, id_attr):
                                        doc_data["id"] = getattr(doc, id_attr)
                                        break
                                
                                # Try different attribute names for status
                                for status_attr in ["status", "state", "processing_status"]:
                                    if hasattr(doc, status_attr):
                                        doc_data["status"] = getattr(doc, status_attr)
                                        break
                                
                                # Try different attribute names for metadata
                                for meta_attr in ["metadata", "meta", "properties"]:
                                    if hasattr(doc, meta_attr):
                                        doc_data["metadata"] = getattr(doc, meta_attr) or {}
                                        break
                                
                                # Ensure we have at least these keys with defaults
                                if "id" not in doc_data:
                                    doc_data["id"] = "unknown"
                                if "status" not in doc_data:
                                    doc_data["status"] = "unknown"
                                if "metadata" not in doc_data:
                                    doc_data["metadata"] = {}
                                    
                                documents.append(doc_data)
                            except Exception as e:
                                logger.error(f"Error processing document from page {current_page+1}: {e}")
                    elif hasattr(next_response, "__iter__"):
                        for doc in next_response:
                            try:
                                # Extract document data with a more flexible approach
                                doc_data = {}
                                
                                # Try different attribute names for ID
                                for id_attr in ["id", "document_id", "_id", "uuid"]:
                                    if hasattr(doc, id_attr):
                                        doc_data["id"] = getattr(doc, id_attr)
                                        break
                                
                                # Try different attribute names for status
                                for status_attr in ["status", "state", "processing_status"]:
                                    if hasattr(doc, status_attr):
                                        doc_data["status"] = getattr(doc, status_attr)
                                        break
                                
                                # Try different attribute names for metadata
                                for meta_attr in ["metadata", "meta", "properties"]:
                                    if hasattr(doc, meta_attr):
                                        doc_data["metadata"] = getattr(doc, meta_attr) or {}
                                        break
                                
                                # Ensure we have at least these keys with defaults
                                if "id" not in doc_data:
                                    doc_data["id"] = "unknown"
                                if "status" not in doc_data:
                                    doc_data["status"] = "unknown"
                                if "metadata" not in doc_data:
                                    doc_data["metadata"] = {}
                                    
                                documents.append(doc_data)
                            except Exception as e:
                                logger.error(f"Error processing document in page iteration: {e}")
                    
                    # Update cursor for next page
                    cursor = getattr(next_response, "cursor", None)
                    if not cursor:
                        break
                    
                    current_page += 1
                    
                except Exception as e:
                    logger.error(f"Error fetching page {current_page+1}: {e}")
                    break
            
            logger.info(f"Retrieved {len(documents)} documents from Ragie")
            return documents
            
        except SDKError as e:
            logger.error(f"Error listing documents: {e}")
            raise
            
    def get_document_chunks(self, 
                          document_id: str, 
                          start_index: Optional[int] = None,
                          end_index: Optional[int] = None,
                          page_size: int = 10,
                          cursor: Optional[str] = None,
                          partition: Optional[str] = None) -> Dict[str, Any]:
        """
        Get chunks for a document.
        
        Args:
            document_id: The ID of the document
            start_index: Optional starting index for chunks
            end_index: Optional ending index for chunks
            page_size: Number of chunks per page (1-100)
            cursor: Optional cursor for pagination
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            Dictionary with chunks and pagination info
        """
        try:
            # Build query parameters
            params = {}
            if start_index is not None:
                params["start_index"] = start_index
            if end_index is not None:
                params["end_index"] = end_index
            if cursor:
                params["cursor"] = cursor
            
            # Ensure page_size is within bounds (1-100)
            page_size = max(1, min(100, page_size))
            params["page_size"] = page_size
            
            # Get the partition to use
            partition_val = partition or self.default_partition
            
            # Call the Ragie API
            logger.info(f"Getting chunks for document {document_id} with params: {params}")
            
            # Note: Don't pass headers parameter as it's not supported in this version
            response = self.client.documents.get_chunks(document_id, **params)
            
            # Process the response
            chunks = []
            next_cursor = None
            
            # Extract chunks from the response
            if hasattr(response, "chunks"):
                chunks = response.chunks
            elif hasattr(response, "items"):
                chunks = response.items
            elif hasattr(response, "result"):
                if hasattr(response.result, "chunks"):
                    chunks = response.result.chunks
                elif hasattr(response.result, "items"):
                    chunks = response.result.items
            
            # Extract next cursor for pagination
            if hasattr(response, "cursor"):
                next_cursor = response.cursor
            elif hasattr(response, "next_cursor"):
                next_cursor = response.next_cursor
            elif hasattr(response, "result") and hasattr(response.result, "cursor"):
                next_cursor = response.result.cursor
            
            # Format chunks for standardized return format
            formatted_chunks = []
            for chunk in chunks:
                try:
                    chunk_data = {
                        "id": getattr(chunk, "id", None),
                        "index": getattr(chunk, "index", -1),
                        "text": getattr(chunk, "text", ""),
                        "metadata": getattr(chunk, "metadata", {}) or {},
                        "document_id": document_id
                    }
                    formatted_chunks.append(chunk_data)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
            
            return {
                "chunks": formatted_chunks,
                "next_cursor": next_cursor
            }
            
        except SDKError as e:
            logger.error(f"Error getting document chunks: {e}")
            raise

    def get_document_chunk(self, 
                         document_id: str, 
                         chunk_id: str,
                         partition: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a specific document chunk by its ID.
        
        Args:
            document_id: The ID of the document
            chunk_id: The ID of the chunk
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            Dict containing chunk data
        """
        try:
            # Get the partition to use but don't pass it as headers
            partition_val = partition or self.default_partition
            
            # Call the Ragie API without headers parameter
            logger.info(f"Getting chunk {chunk_id} for document {document_id}")
            
            # SDK v1.5.0 doesn't support headers parameter
            response = self.client.documents.get_chunk(document_id, chunk_id)
            
            # Process the response into a consistent format
            chunk_data = {
                "id": getattr(response, "id", chunk_id),
                "index": getattr(response, "index", -1),
                "text": getattr(response, "text", "") or getattr(response, "content", ""),
                "metadata": getattr(response, "metadata", {})
            }
            
            return chunk_data
        except SDKError as e:
            logger.error(f"Error getting document chunk: {e}")
            raise

    def get_document_content(self, 
                           document_id: str,
                           partition: Optional[str] = None) -> str:
        """
        Get the content of a document.
        
        Args:
            document_id: The ID of the document
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            The document content as a string
        """
        try:
            # Get the partition to use but don't pass it as headers
            partition_val = partition or self.default_partition
            
            # Call the Ragie API without headers parameter
            logger.info(f"Getting content for document {document_id}")
            response = self.client.documents.get_content(document_id)
            
            # Extract content based on response structure
            content = ""
            
            # Try multiple possible attribute names
            for attr_name in ["content", "text", "data"]:
                if hasattr(response, attr_name) and getattr(response, attr_name):
                    content = getattr(response, attr_name)
                    break
            
            return content
        except SDKError as e:
            logger.error(f"Error getting document content: {e}")
            raise

    def get_document_source(self, 
                          document_id: str,
                          partition: Optional[str] = None) -> bytes:
        """
        Get the source file of a document.
        
        Args:
            document_id: The ID of the document
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            Bytes containing the original source file
        """
        try:
            # Get partition value but don't pass it as headers
            partition_val = partition or self.default_partition
            
            # Call the Ragie API without headers parameter
            logger.info(f"Getting source file for document {document_id}")
            response = self.client.documents.get_source(document_id)
            
            # Handle different response types
            if hasattr(response, "content") and isinstance(response.content, bytes):
                return response.content
            elif hasattr(response, "source") and isinstance(response.source, bytes):
                return response.source
            else:
                # Try to get bytes from response directly
                return bytes(response)
        except SDKError as e:
            logger.error(f"Error getting document source: {e}")
            raise

    def get_document_summary(self,
                           document_id: str,
                           partition: Optional[str] = None) -> Dict[str, Any]:
        """
        Get an AI-generated summary of a document.
        
        Args:
            document_id: The ID of the document
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            Dictionary with summary text and metadata
        """
        try:
            # Get the partition to use but don't pass it as headers
            partition_val = partition or self.default_partition
            
            # Call the Ragie API without headers parameter
            logger.info(f"Getting summary for document {document_id}")
            response = self.client.documents.get_summary(document_id)
            
            # Extract summary based on response structure
            summary = {}
            
            # First try to get the summary text
            summary_text = ""
            for attr_name in ["summary", "text", "content"]:
                if hasattr(response, attr_name) and getattr(response, attr_name):
                    summary_text = getattr(response, attr_name)
                    break
                    
            summary["text"] = summary_text
            
            # Try to get any metadata
            if hasattr(response, "metadata"):
                summary["metadata"] = response.metadata or {}
            else:
                summary["metadata"] = {}
                
            return summary
        except SDKError as e:
            logger.error(f"Error getting document summary: {e}")
            raise

    def get_document(self, document_id: str, partition: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a document.
        
        Args:
            document_id: The ID of the document
            partition: Optional partition identifier (defaults to self.default_partition)
            
        Returns:
            Dict containing document information
        """
        try:
            # Get the partition to use but don't pass it as header
            partition_val = partition or self.default_partition
            
            # Call the Ragie API without headers parameter
            logger.info(f"Getting document info for {document_id}")
            response = self.client.documents.get(document_id)
            
            # Extract document information
            doc_data = {
                "id": getattr(response, "id", document_id),
                "status": getattr(response, "status", "unknown"),
                "metadata": getattr(response, "metadata", {}) or {}
            }
            
            # Add creation and update times if available
            if hasattr(response, "created_at"):
                doc_data["created_at"] = response.created_at
            if hasattr(response, "updated_at"):
                doc_data["updated_at"] = response.updated_at
                
            return doc_data
        except SDKError as e:
            logger.error(f"Error getting document info: {e}")
            raise 