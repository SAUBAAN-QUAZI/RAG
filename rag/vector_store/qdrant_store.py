"""
Qdrant Vector Store Implementation
---------------------------------
This module implements a vector store using Qdrant Cloud for efficient similarity search.
"""

import os
import uuid
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union

# Import only the models, not the full client
# This avoids the problematic local dependencies
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams, Filter

from rag.config import (
    EMBEDDING_MODEL,
    VECTOR_DB_API_KEY,
    VECTOR_DB_URL,
    VECTOR_DB_PATH
)
from rag.document_processing.document import DocumentChunk
from rag.utils import logger
from rag.vector_store.base import VectorStore


class QdrantStore(VectorStore):
    """
    Vector store implementation using Qdrant Cloud.
    
    This implementation uses direct REST API calls to avoid dependency issues.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_collection",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        local_path: Optional[Union[str, Path]] = None,  # Kept for compatibility
        vector_size: int = 1536,  # Default for OpenAI embeddings
    ):
        """
        Initialize a QdrantStore using Qdrant Cloud.
        
        Args:
            collection_name: Name of the collection to use
            url: URL of Qdrant service
            api_key: API key for Qdrant service
            local_path: Not used in cloud-only version
            vector_size: Size of embedding vectors
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Maintain a mapping of string IDs to UUIDs
        self.id_mapping = {}
        
        # Performance metrics
        self.metrics = {
            "search_count": 0,
            "search_errors": 0,
            "search_latency_ms": [],
            "request_timeouts": 0,
            "connection_errors": 0
        }
        
        # Default timeout values (in seconds)
        self.default_timeout = 10.0  # For normal operations
        self.search_timeout = 5.0    # For search operations (more time-sensitive)
        self.upload_timeout = 20.0   # For batch uploads (can take longer)
        
        # Determine embedding model dimension
        if EMBEDDING_MODEL == "text-embedding-3-small":
            self.vector_size = 1536
        elif EMBEDDING_MODEL == "text-embedding-3-large":
            self.vector_size = 3072
        elif EMBEDDING_MODEL == "text-embedding-ada-002":
            self.vector_size = 1536
            
        # Get cloud credentials
        self.url = url or VECTOR_DB_URL
        self.api_key = api_key or VECTOR_DB_API_KEY
        
        if not self.url:
            raise ValueError(
                "Qdrant Cloud URL not provided. Please set VECTOR_DB_URL in your .env file."
            )
            
        if not self.api_key:
            raise ValueError(
                "Qdrant Cloud API key not provided. Please set VECTOR_DB_API_KEY in your .env file."
            )
            
        # Format the URL properly
        if self.url.endswith('/'):
            self.url = self.url[:-1]
            
        # Use REST port if not specified
        if ':6333' not in self.url and ':443' not in self.url:
            self.url = f"{self.url}:6333"
            
        self.is_local = False
        
        # Create HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "api-key": self.api_key
        })
        
        logger.info(f"Connecting to Qdrant cloud at {self.url}")
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
        logger.info(f"Initialized QdrantStore with collection={collection_name}")
    
    def _make_request(self, method, endpoint, data=None, params=None, timeout=None):
        """
        Make a request to the Qdrant API with timing, retry logic and detailed logging.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request data
            params: Query parameters
            timeout: Request timeout in seconds (if None, uses defaults)
            
        Returns:
            dict: API response
        """
        url = f"{self.url}/{endpoint}"
        
        # Use appropriate timeout based on operation type or default
        if timeout is None:
            timeout = self.default_timeout
        
        # Log request details
        log_msg = f"Making {method} request to {endpoint}"
        if params:
            log_msg += f" with params: {params}"
        if data and not isinstance(data, dict):
            log_msg += f" with data type: {type(data)}"
        elif data:
            # Only log basic data information, not full payload which could be large
            data_keys = list(data.keys()) if isinstance(data, dict) else "None"
            data_size = len(json.dumps(data))
            log_msg += f" with data keys: {data_keys}, size: {data_size} bytes"
            
        logger.debug(log_msg)
        
        # Track request timing
        start_time = time.time()
        
        # Maximum retries
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                if method == 'GET':
                    response = self.session.get(url, params=params, timeout=timeout)
                elif method == 'POST':
                    response = self.session.post(url, json=data, params=params, timeout=timeout)
                elif method == 'PUT':
                    response = self.session.put(url, json=data, timeout=timeout)
                elif method == 'DELETE':
                    response = self.session.delete(url, timeout=timeout)
                    
                response.raise_for_status()
                
                # Calculate request duration
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(f"Request to {endpoint} completed in {duration_ms:.2f}ms")
                
                if response.status_code == 204:  # No content
                    return None
                
                result = response.json()
                
                # Log result summary (not full result to avoid flooding logs)
                if isinstance(result, dict) and 'result' in result:
                    result_type = type(result['result']).__name__
                    result_size = len(str(result['result']))
                    logger.debug(f"Received result of type {result_type}, size {result_size} bytes")
                
                return result
                
            except requests.exceptions.Timeout:
                retry_count += 1
                self.metrics["request_timeouts"] += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                last_error = f"Request timeout (attempt {retry_count}/{max_retries})"
                logger.warning(f"{last_error}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except requests.exceptions.ConnectionError as e:
                retry_count += 1
                self.metrics["connection_errors"] += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                last_error = f"Connection error: {str(e)}"
                logger.warning(f"{last_error}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except Exception as e:
                # Non-retryable error
                logger.error(f"Error making request to Qdrant API: {str(e)}")
                raise
        
        # If we've exhausted retries, raise the last error
        logger.error(f"Error making request to Qdrant API after {max_retries} retries: {last_error}")
        raise requests.exceptions.RequestException(last_error)
    
    def _create_collection_if_not_exists(self) -> None:
        """
        Create the collection if it doesn't exist.
        """
        try:
            # Get collections
            result = self._make_request('GET', 'collections')
            collection_names = [collection['name'] for collection in result.get('result', {}).get('collections', [])]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating new Qdrant collection: {self.collection_name}")
                
                # Create collection with optimized parameters
                data = {
                    "vectors": {
                        "size": self.vector_size,
                        "distance": "Cosine"
                    },
                    # Optional optimization parameters
                    "optimizers_config": {
                        "indexing_threshold": 20000,  # Start indexing after this many vectors
                    },
                    "hnsw_config": {
                        "m": 16,  # Higher means more accurate but slower
                        "ef_construct": 100,  # Higher means more accurate index but slower to build
                    }
                }
                
                self._make_request('PUT', f'collections/{self.collection_name}', data=data, timeout=15.0)
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error creating/accessing Qdrant collection: {e}")
            raise
    
    def _string_id_to_uuid(self, string_id: str) -> str:
        """
        Convert a string ID to a UUID for Qdrant compatibility.
        
        Args:
            string_id: String ID to convert
            
        Returns:
            str: UUID string
        """
        # If we've seen this ID before, return the cached UUID
        if string_id in self.id_mapping:
            return self.id_mapping[string_id]
        
        # Generate a deterministic UUID from the string ID
        # This ensures the same string ID always maps to the same UUID
        namespace = uuid.UUID('00000000-0000-0000-0000-000000000000')
        new_uuid = str(uuid.uuid5(namespace, string_id))
        
        # Cache the mapping
        self.id_mapping[string_id] = new_uuid
        
        return new_uuid
    
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
            
        # Prepare points for Qdrant
        points = []
        
        for chunk in chunks:
            chunk_id = chunk.chunk_id
            
            # Skip if embedding not available
            if chunk_id not in embeddings:
                logger.warning(f"No embedding found for chunk {chunk_id}")
                continue
                
            # Convert string ID to UUID for Qdrant compatibility
            qdrant_id = self._string_id_to_uuid(chunk_id)
            
            # Store the original ID in metadata for retrieval
            metadata = {**chunk.metadata, "original_id": chunk_id}
            
            # Prepare point
            points.append({
                "id": qdrant_id,
                "vector": embeddings[chunk_id],
                "payload": {
                    "content": chunk.content,
                    **metadata,  # Include original ID in metadata
                }
            })
            
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self._make_request(
                    'PUT', 
                    f'collections/{self.collection_name}/points', 
                    data={"points": batch},
                    timeout=self.upload_timeout  # Use longer timeout for uploads
                )
                logger.info(f"Added batch of {len(batch)} points to Qdrant")
            except Exception as e:
                logger.error(f"Error adding batch to Qdrant: {str(e)}")
                # Continue with next batch
                continue
            
        logger.info(f"Added {len(points)} embeddings to Qdrant collection")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search for similar vectors in the store with optimized parameters.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List of dictionaries containing chunk information and similarity scores
        """
        try:
            # Start timing
            start_time = time.time()
            self.metrics["search_count"] += 1
            
            # Cap top_k to reasonable limits
            if top_k > 20:
                logger.warning(f"Requested top_k={top_k} is too high, capping at 20")
                top_k = 20
            elif top_k < 1:
                logger.warning(f"Invalid top_k value: {top_k}, using default of 5")
                top_k = 5
                
            # Log search parameters
            logger.info(f"Searching vector store with top_k={top_k}, filters={filter_dict}")
            
            # Create optimized search request
            search_request = {
                "vector": query_embedding,
                "limit": top_k,
                "with_payload": True,
                # Optimized search parameters
                "params": {
                    "ef": 128,  # Higher value means more accurate but slower search
                    "hnsw_ef": 128,  # For better search quality
                }
            }
            
            # Add filter if needed
            if filter_dict:
                # Convert filter dictionary to Qdrant filter
                filter_conditions = []
                
                for key, value in filter_dict.items():
                    if isinstance(value, list):
                        # Handle list values (any match)
                        should_conditions = []
                        for val in value:
                            should_conditions.append({
                                "key": key,
                                "match": {"value": val}
                            })
                        filter_conditions.append({"should": should_conditions})
                    else:
                        # Handle single value (exact match)
                        filter_conditions.append({
                            "key": key,
                            "match": {"value": value}
                        })
                
                search_request["filter"] = {"must": filter_conditions}
            
            # Call API with shorter timeout for search operations
            result = self._make_request(
                'POST', 
                f'collections/{self.collection_name}/points/search', 
                data=search_request,
                timeout=self.search_timeout
            )
            
            # Format results
            formatted_results = []
            
            if not result or not result.get('result'):
                logger.warning("Search returned no results")
                return []
            
            # Process response
            search_results = result.get('result', [])
            
            # Log search results
            logger.info(f"Search returned {len(search_results)} matches")
            
            for result in search_results:
                # Extract content and metadata from payload
                payload = result.get('payload', {})
                content = payload.pop("content", "")
                
                # Get the original ID from metadata if available
                original_id = payload.pop("original_id", result.get('id'))
                
                # Only include metadata that is useful for search/filtering
                # This reduces the size of the response
                filtered_metadata = {}
                for key, value in payload.items():
                    if key in ["title", "source", "author", "page", "timestamp", "category"]:
                        filtered_metadata[key] = value
                
                formatted_result = {
                    "chunk_id": original_id,  # Use original string ID
                    "content": content,
                    "metadata": filtered_metadata,
                    "similarity": result.get('score', 0),
                }
                formatted_results.append(formatted_result)
            
            # Calculate and record search latency
            duration_ms = (time.time() - start_time) * 1000
            self.metrics["search_latency_ms"].append(duration_ms)
            
            logger.info(f"Found {len(formatted_results)} results for query in {duration_ms:.2f}ms")
            return formatted_results
            
        except Exception as e:
            # Record search error
            self.metrics["search_errors"] += 1
            
            # Log error with detailed information
            logger.exception(f"Error during vector search: {str(e)}")
            
            # Return empty results in case of error
            return []
            
    def get_metrics(self) -> Dict:
        """
        Get performance metrics for the vector store.
        
        Returns:
            Dict: Performance metrics
        """
        # Calculate average search latency
        if self.metrics["search_latency_ms"]:
            avg_latency = sum(self.metrics["search_latency_ms"]) / len(self.metrics["search_latency_ms"])
        else:
            avg_latency = 0
            
        # Calculate error rate
        if self.metrics["search_count"] > 0:
            error_rate = (self.metrics["search_errors"] / self.metrics["search_count"]) * 100
        else:
            error_rate = 0
            
        # Prepare metrics report
        metrics_report = {
            "search_count": self.metrics["search_count"],
            "search_errors": self.metrics["search_errors"],
            "error_rate": f"{error_rate:.2f}%",
            "avg_search_latency_ms": f"{avg_latency:.2f}",
            "request_timeouts": self.metrics["request_timeouts"],
            "connection_errors": self.metrics["connection_errors"],
        }
        
        return metrics_report
    
    def save(self, file_path: str = None) -> None:
        """
        Save the vector store to disk.
        
        For cloud deployments, this is a no-op.
        
        Args:
            file_path: Not used for Qdrant Cloud
        """
        logger.info("Qdrant Cloud automatically persists data, no explicit save needed")
    
    @classmethod
    def load(cls, file_path: str = None, collection_name: str = "rag_collection") -> "QdrantStore":
        """
        Load a vector store from disk or connect to cloud.
        
        Args:
            file_path: Not used for Qdrant Cloud
            collection_name: Name of the collection to load
            
        Returns:
            QdrantStore: Loaded vector store
        """
        return cls(collection_name=collection_name)
    
    def clear(self) -> None:
        """
        Clear all vectors from the store.
        """
        try:
            # Delete collection
            self._make_request('DELETE', f'collections/{self.collection_name}')
            
            # Recreate the collection
            self._create_collection_if_not_exists()
            logger.info(f"Cleared all vectors from collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict: Statistics about the vector store
        """
        try:
            result = self._make_request('GET', f'collections/{self.collection_name}')
            collection_info = result.get('result', {})
            
            stats = {
                "collection_name": self.collection_name,
                "vector_count": collection_info.get('vectors_count', 0),
                "is_local": False,
                "vector_dimension": self.vector_size,
            }
            
            # Add performance metrics
            stats["performance"] = self.get_metrics()
            
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "vector_count": 0,
                "is_local": False, 
                "error": str(e),
            } 