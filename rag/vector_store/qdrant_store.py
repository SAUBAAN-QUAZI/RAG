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
from typing import Dict, List, Optional, Union, Any
import numpy as np

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
    Optimized for high-performance similarity search with advanced search parameters.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_collection",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        local_path: Optional[Union[str, Path]] = None,  # Kept for compatibility
        vector_size: int = 1536,  # Default for OpenAI embeddings
        distance_metric: str = "Cosine",  # Cosine, Euclid, or Dot
        optimize_index: bool = True,
        search_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a QdrantStore using Qdrant Cloud with optimized parameters.
        
        Args:
            collection_name: Name of the collection to use
            url: URL of Qdrant service
            api_key: API key for Qdrant service
            local_path: Not used in cloud-only version
            vector_size: Size of embedding vectors
            distance_metric: Distance metric for similarity calculation
            optimize_index: Whether to optimize the index for performance
            search_params: Optional parameters for search customization
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        self.optimize_index = optimize_index
        
        # Default optimized search parameters
        self.default_search_params = {
            "ef": 128,  # Higher values improve recall but increase latency
            "ef_construct": 100,  # Higher values improve index quality but increase build time
            "m": 16,  # Number of bi-directional links created for each element (higher = better recall, lower = faster)
            "hnsw_ef": 128,  # Query time ef parameter (higher = more accurate)
            "exact": False,  # Whether to use exact search (accurate but slow)
            "quantization": False  # Whether to use scalar quantization (faster but less accurate)
        }
        
        # Apply custom search params if provided
        if search_params:
            self.search_params = {**self.default_search_params, **search_params}
        else:
            self.search_params = self.default_search_params
        
        # Optimized timeouts based on common operations
        self.default_timeout = 5.0  # Default timeout for most operations
        self.upload_timeout = 20.0  # Longer timeout for uploads
        self.index_timeout = 30.0   # Even longer timeout for indexing
        
        # Maintain a mapping of string IDs to UUIDs
        self.id_mapping = {}
        
        # Performance metrics
        self.metrics = {
            "search_count": 0,
            "search_errors": 0,
            "search_latency_ms": [],
            "request_timeouts": 0,
            "connection_errors": 0,
            "avg_result_count": 0
        }
        
        # Set the URL from params or config
        self.url = url or VECTOR_DB_URL
        if not self.url:
            self.url = "http://localhost:6333"  # Fall back to local
            logger.warning(f"No Qdrant URL provided, using local: {self.url}")
        else:
            # Remove any trailing /v1 from the URL
            if self.url.endswith('/v1'):
                self.url = self.url[:-3]
                logger.warning(f"Removed '/v1' suffix from URL: {self.url}")
            elif self.url.endswith('/v1/'):
                self.url = self.url[:-4]
                logger.warning(f"Removed '/v1/' suffix from URL: {self.url}")
            
            logger.info(f"Using Qdrant at: {self.url}")
            
        # Set API key from params or config
        self.api_key = api_key or VECTOR_DB_API_KEY
        if self.api_key:
            logger.info("Using Qdrant API key")
            
        # Create collection if it doesn't exist
        try:
            self._create_collection_if_not_exists()
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {str(e)}")
            # Continue anyway - the collection might already exist or be created later
            
        logger.info(f"Initialized QdrantStore with collection={collection_name}, distance={distance_metric}, "
                  f"vector_size={vector_size}, optimize_index={optimize_index}")
    
    def _string_id_to_uuid(self, string_id: str) -> str:
        """
        Convert a string ID to a UUID for Qdrant compatibility.
        
        Args:
            string_id: String ID
            
        Returns:
            UUID string generated from the string ID
        """
        # Check if we already have this ID mapped
        if string_id in self.id_mapping:
            return self.id_mapping[string_id]
            
        # Generate a deterministic UUID from the string ID
        try:
            # Try to use the string as a UUID if it's already in that format
            uuid_obj = uuid.UUID(string_id)
            uuid_str = str(uuid_obj)
        except ValueError:
            # Otherwise, create a namespace UUID
            # Use a fixed namespace and the string ID to generate a deterministic UUID
            namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # UUID namespace for URLs
            uuid_str = str(uuid.uuid5(namespace, string_id))
            
        # Store the mapping
        self.id_mapping[string_id] = uuid_str
        return uuid_str
    
    def _uuid_to_string_id(self, uuid_str: str) -> Optional[str]:
        """
        Convert a UUID back to the original string ID.
        
        Args:
            uuid_str: UUID string
            
        Returns:
            Original string ID or None if not found
        """
        # Search the mapping for the UUID
        for string_id, mapped_uuid in self.id_mapping.items():
            if mapped_uuid == uuid_str:
                return string_id
        return None
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        timeout: float = None,
        max_retries: int = 3,
    ) -> Optional[Dict]:
        """
        Make a request to the Qdrant API with optimized retry handling.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request data
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for transient errors
            
        Returns:
            Response data or None
        """
        # Use default timeout if not specified
        timeout = timeout or self.default_timeout
        
        # Remove v1 prefix if present in the endpoint
        if endpoint.startswith('v1/'):
            endpoint = endpoint[3:]
            logger.info(f"Removing 'v1/' prefix from endpoint: {endpoint}")
        
        url = f"{self.url}/{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        # Add API key if available
        if self.api_key:
            headers["api-key"] = self.api_key
            
        # Prepare request arguments
        request_args = {
            "headers": headers,
            "timeout": timeout,
        }
        
        # Add data if provided
        if data is not None:
            request_args["json"] = data
            
        # Initialize retry variables
        retry_count = 0
        last_error = None
        
        # Log the endpoint being requested
        logger.debug(f"Making {method} request to endpoint: {url}")
        
        # Retry loop
        while retry_count <= max_retries:
            try:
                # Make request
                if method == "GET":
                    response = requests.get(url, **request_args)
                elif method == "POST":
                    response = requests.post(url, **request_args)
                elif method == "PUT":
                    response = requests.put(url, **request_args)
                elif method == "DELETE":
                    response = requests.delete(url, **request_args)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Handle errors
                if response.status_code >= 400:
                    error_message = f"Error from Qdrant API: {response.status_code} - {response.text}"
                    logger.error(error_message)
                    
                    # Return None for 404s (not found)
                    if response.status_code == 404:
                        return None
                        
                    # Retry on server errors (5xx)
                    if response.status_code >= 500:
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = 2 ** retry_count  # Exponential backoff
                            logger.warning(f"Server error, retrying in {wait_time}s ({retry_count}/{max_retries})...")
                            time.sleep(wait_time)
                            continue
                            
                    # Raise for other errors
                    response.raise_for_status()
                
                # Handle success
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
        Create the collection if it doesn't exist with optimized settings.
        """
        try:
            # Get collections
            result = self._make_request('GET', 'collections')
            
            # Handle empty or None response
            if not result:
                logger.warning("Empty response when getting collections")
                collections_exist = False
                collection_names = []
            else:
                collections_data = result.get('result', {})
                collections = collections_data.get('collections', [])
                
                # Handle different response structures
                if isinstance(collections, list):
                    collection_names = [collection.get('name') for collection in collections]
                else:
                    logger.warning(f"Unexpected collections structure: {collections}")
                    collection_names = []
                
                collections_exist = bool(collection_names)
                logger.info(f"Found existing collections: {collection_names}")
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating new Qdrant collection: {self.collection_name}")
                
                # Create collection with optimized parameters
                data = {
                    "vectors": {
                        "size": self.vector_size,
                        "distance": self.distance_metric
                    },
                    # Optimized configuration parameters
                    "optimizers_config": {
                        "indexing_threshold": 5000,  # Start indexing after this many vectors (lower for faster indexing)
                        "memmap_threshold": 50000,  # When to start using memmap for vector storage
                        "payload_indexing": {
                            "metadata": {
                                "min_word_length": 2,  # Minimum word length to index (for faster exact matching)
                                "max_total_tokens": 5000  # Maximum tokens to index per collection
                            }
                        }
                    },
                    # HNSW configuration
                    "hnsw_config": {
                        "m": self.search_params["m"],  # Number of bidirectional links (16 is a good default)
                        "ef_construct": self.search_params["ef_construct"],  # Number of candidates to consider during index building
                        "on_disk": False,  # Whether to store HNSW index on disk (false for better performance)
                        "full_scan_threshold": 10000  # When to use full scan instead of HNSW
                    },
                    # Quantization settings for larger collections
                    "quantization_config": {
                        "scalar": {
                            "type": "int8",  # 8-bit quantization for good balance of speed and accuracy
                            "always_ram": True  # Keep quantized vectors in RAM for faster access
                        }
                    } if self.search_params["quantization"] else None
                }
                
                # Remove None values
                data = {k: v for k, v in data.items() if v is not None}
                
                # Create collection with increased timeout
                self._make_request('PUT', f'collections/{self.collection_name}', data=data, timeout=self.index_timeout)
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
                # If collection exists and optimize_index is True, update index params
                if self.optimize_index:
                    logger.info(f"Optimizing existing collection: {self.collection_name}")
                    # Update HNSW parameters
                    hnsw_config = {
                        "m": self.search_params["m"],
                        "ef_construct": self.search_params["ef_construct"]
                    }
                    self._make_request('PUT', f'collections/{self.collection_name}/config', data={"hnsw_config": hnsw_config})
                
        except Exception as e:
            logger.error(f"Error checking/creating collection: {str(e)}")
            raise
    
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
            
        logger.info(f"Adding {len(chunks)} embeddings to Qdrant collection")
        
        # Prepare points
        points = []
        
        for chunk in chunks:
            # Get chunk ID
            chunk_id = chunk.metadata.get("chunk_id", str(uuid.uuid4()))
            
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
            
        # Add to collection in batches with optimized batch size
        batch_size = 100  # Optimal batch size for performance
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
        
        # If we've added a significant number of vectors, optimize the index
        if len(points) > 1000 and self.optimize_index:
            try:
                logger.info(f"Optimizing index after adding {len(points)} vectors")
                self._make_request(
                    'POST',
                    f'collections/{self.collection_name}/index',
                    timeout=self.index_timeout
                )
            except Exception as e:
                logger.warning(f"Error optimizing index: {str(e)}")
            
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
            if top_k > 30:
                logger.warning(f"Requested top_k={top_k} is too high, capping at 30")
                top_k = 30
            elif top_k < 1:
                logger.warning(f"Invalid top_k value: {top_k}, using default of 5")
                top_k = 5
                
            # Increase the search limit by 50% to ensure we get enough results after filtering
            search_limit = min(int(top_k * 1.5), 50)
                
            # Log search parameters
            logger.info(f"Searching vector store with top_k={top_k}, search_limit={search_limit}, filters={filter_dict}")
            
            # Create optimized search request
            search_request = {
                "vector": query_embedding,
                "limit": search_limit,
                "with_payload": True,
                # Optimized search parameters
                "params": {
                    "ef": self.search_params["ef"],  # Higher value means more accurate but slower search
                    "hnsw_ef": self.search_params["hnsw_ef"],  # For better search quality
                    "exact": self.search_params["exact"]  # Whether to use exact search
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
                        # Handle single values (exact match)
                        filter_conditions.append({
                            "key": key,
                            "match": {"value": value}
                        })
                
                # Add filter to search request
                if filter_conditions:
                    search_request["filter"] = {"must": filter_conditions}
            
            # Make search request
            response = self._make_request(
                'POST',
                f'collections/{self.collection_name}/points/search',
                data=search_request,
                timeout=self.default_timeout
            )
            
            # Calculate search time and update metrics
            search_time_ms = (time.time() - start_time) * 1000
            self.metrics["search_latency_ms"].append(search_time_ms)
            
            # Process results
            if response and 'result' in response:
                results = response['result']
                self.metrics["avg_result_count"] = (
                    (self.metrics["avg_result_count"] * (self.metrics["search_count"] - 1) + len(results)) 
                    / self.metrics["search_count"]
                )
                
                # Convert to standard format
                formatted_results = []
                
                for item in results:
                    # Get original string ID if possible
                    point_id = item.get('id')
                    chunk_id = self._uuid_to_string_id(point_id) if point_id else None
                    
                    # Get payload (contains content and metadata)
                    payload = item.get('payload', {})
                    
                    # Extract content and metadata
                    content = payload.pop('content', None)
                    metadata = payload
                    
                    # Get similarity score (1 - distance for cosine)
                    score = item.get('score', 0)
                    
                    # For cosine distance, convert score to similarity (1 - distance)
                    if self.distance_metric == "Cosine":
                        similarity = float(score)
                    else:
                        # For other distances (like Euclidean), normalize to 0-1 range
                        # using a simple exponential decay function
                        similarity = float(max(0, min(1, np.exp(-score))))
                    
                    # Format result
                    result = {
                        "chunk_id": chunk_id or point_id,  # Fallback to point_id if chunk_id not available
                        "content": content,
                        "metadata": metadata,
                        "similarity": similarity
                    }
                    
                    formatted_results.append(result)
                    
                # Limit to requested top_k
                formatted_results = formatted_results[:top_k]
                
                logger.info(f"Search completed in {search_time_ms:.2f}ms, found {len(formatted_results)} results")
                return formatted_results
            
            logger.warning("No results found in search response")
            return []
            
        except Exception as e:
            self.metrics["search_errors"] += 1
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """
        Delete chunks from the vector store.
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        if not chunk_ids:
            logger.warning("No chunk IDs to delete")
            return
            
        logger.info(f"Deleting {len(chunk_ids)} chunks from vector store")
        
        # Convert string IDs to UUIDs
        point_ids = [self._string_id_to_uuid(chunk_id) for chunk_id in chunk_ids]
        
        # Delete in batches
        batch_size = 100
        for i in range(0, len(point_ids), batch_size):
            batch = point_ids[i:i + batch_size]
            try:
                self._make_request(
                    'POST',
                    f'collections/{self.collection_name}/points/delete',
                    data={"points": batch},
                    timeout=self.default_timeout
                )
                logger.info(f"Deleted batch of {len(batch)} points from Qdrant")
            except Exception as e:
                logger.error(f"Error deleting batch from Qdrant: {str(e)}")
                # Continue with next batch
                continue
                
        # Remove deleted IDs from mapping
        for chunk_id in chunk_ids:
            if chunk_id in self.id_mapping:
                del self.id_mapping[chunk_id]
                
        logger.info(f"Deleted {len(chunk_ids)} chunks from vector store")
    
    def clear(self) -> None:
        """
        Clear the vector store.
        """
        logger.info(f"Clearing vector store collection: {self.collection_name}")
        
        try:
            # Delete collection
            self._make_request(
                'DELETE',
                f'collections/{self.collection_name}',
                timeout=self.default_timeout
            )
            
            # Recreate collection
            self._create_collection_if_not_exists()
            
            # Clear ID mapping
            self.id_mapping = {}
            
            logger.info(f"Cleared vector store collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict: Collection statistics
        """
        try:
            # Get collection info
            response = self._make_request(
                'GET',
                f'collections/{self.collection_name}',
                timeout=self.default_timeout
            )
            
            # Get collection metrics
            metrics_response = self._make_request(
                'GET',
                f'collections/{self.collection_name}/metrics',
                timeout=self.default_timeout
            )
            
            # Combine results
            stats = {}
            
            if response and 'result' in response:
                stats.update(response['result'])
                
            if metrics_response and 'result' in metrics_response:
                stats["metrics"] = metrics_response['result']
                
            # Add client-side metrics
            stats["client_metrics"] = {
                "search_count": self.metrics["search_count"],
                "search_errors": self.metrics["search_errors"],
                "avg_search_latency_ms": sum(self.metrics["search_latency_ms"]) / len(self.metrics["search_latency_ms"]) if self.metrics["search_latency_ms"] else 0,
                "request_timeouts": self.metrics["request_timeouts"],
                "connection_errors": self.metrics["connection_errors"],
                "avg_result_count": self.metrics["avg_result_count"]
            }
                
            return stats
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {"error": str(e)}
    
    def optimize_index(self) -> bool:
        """
        Manually optimize the vector index for better performance.
        
        Returns:
            bool: Whether optimization was successful
        """
        logger.info(f"Manually optimizing index for collection: {self.collection_name}")
        
        try:
            # Trigger reindexing
            self._make_request(
                'POST',
                f'collections/{self.collection_name}/index',
                timeout=self.index_timeout
            )
            
            logger.info(f"Successfully optimized index for collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error optimizing index: {str(e)}")
            return False
            
    # Add the following methods to satisfy the VectorStore abstract base class
    
    def save(self, file_path: str) -> None:
        """
        Save the vector store to disk.
        
        For Qdrant Cloud, this saves the ID mapping and configuration to a JSON file.
        The vectors themselves are already stored in the cloud.
        
        Args:
            file_path: Path to save the configuration
        """
        logger.info(f"Saving QdrantStore configuration to {file_path}")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare config to save
            config = {
                "collection_name": self.collection_name,
                "url": self.url,
                "vector_size": self.vector_size,
                "distance_metric": self.distance_metric,
                "id_mapping": self.id_mapping,
                "search_params": self.search_params,
            }
            
            # Save config to JSON file
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Successfully saved QdrantStore configuration to {file_path}")
        except Exception as e:
            logger.error(f"Error saving QdrantStore configuration: {str(e)}")
            raise
            
    @classmethod
    def load(cls, file_path: str) -> "QdrantStore":
        """
        Load a vector store from disk.
        
        For Qdrant Cloud, this loads the ID mapping and configuration from a JSON file.
        
        Args:
            file_path: Path to the saved configuration
            
        Returns:
            QdrantStore: Loaded vector store
        """
        logger.info(f"Loading QdrantStore configuration from {file_path}")
        
        try:
            # Load config from JSON file
            with open(file_path, 'r') as f:
                config = json.load(f)
                
            # Create QdrantStore instance
            store = cls(
                collection_name=config.get("collection_name", "rag_collection"),
                url=config.get("url"),
                vector_size=config.get("vector_size", 1536),
                distance_metric=config.get("distance_metric", "Cosine"),
                search_params=config.get("search_params"),
            )
            
            # Restore ID mapping
            store.id_mapping = config.get("id_mapping", {})
            
            logger.info(f"Successfully loaded QdrantStore configuration from {file_path}")
            return store
        except Exception as e:
            logger.error(f"Error loading QdrantStore configuration: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict: Statistics about the vector store
        """
        # This method is required by the VectorStore abstract base class
        # Reuse our existing get_stats method
        return self.get_stats() 