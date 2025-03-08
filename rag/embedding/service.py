"""
Embedding Service
--------------
This module handles the conversion of text into vector embeddings
using OpenAI's embedding models.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from openai import OpenAI

from rag.config import EMBEDDING_MODEL, OPENAI_API_KEY
from rag.document_processing.document import DocumentChunk
from rag.utils import get_text_hash, logger, retry_with_exponential_backoff, save_json


class EmbeddingService:
    """
    Service for generating text embeddings using OpenAI API.
    """
    
    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = EMBEDDING_MODEL,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
    ):
        """
        Initialize an EmbeddingService.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
            cache_dir: Directory to cache embeddings
            use_cache: Whether to use the embedding cache
        """
        self.api_key = api_key
        self.model = model
        self.use_cache = use_cache
        
        # Set up cache directory
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.use_cache = False
            self.cache_dir = None
            
        # Initialize OpenAI client with minimal configuration
        # Updated for compatibility with different versions of the OpenAI SDK
        try:
            # On Render, remove any proxy settings that might be automatically added
            openai_kwargs = {'api_key': api_key}
            
            # Save original proxy environment variables if they exist
            original_http_proxy = os.environ.pop('http_proxy', None)
            original_https_proxy = os.environ.pop('https_proxy', None)
            original_HTTP_PROXY = os.environ.pop('HTTP_PROXY', None)
            original_HTTPS_PROXY = os.environ.pop('HTTPS_PROXY', None)
            
            # Log proxy detection
            if any([original_http_proxy, original_https_proxy, original_HTTP_PROXY, original_HTTPS_PROXY]):
                logger.info("Proxy settings detected in environment. Temporarily removing for OpenAI client initialization.")
            
            try:
                # Initialize without any proxy settings in environment
                self.client = OpenAI(**openai_kwargs)
                logger.info(f"Successfully initialized OpenAI client")
            finally:
                # Restore original proxy environment variables
                if original_http_proxy:
                    os.environ['http_proxy'] = original_http_proxy
                if original_https_proxy:
                    os.environ['https_proxy'] = original_https_proxy
                if original_HTTP_PROXY:
                    os.environ['HTTP_PROXY'] = original_HTTP_PROXY
                if original_HTTPS_PROXY:
                    os.environ['HTTPS_PROXY'] = original_HTTPS_PROXY
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise
        
        logger.info(f"Initialized EmbeddingService with model={model}, use_cache={use_cache}")
    
    @retry_with_exponential_backoff
    def _get_embeddings_from_api(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
            )
            
            # Extract embeddings from the response
            embeddings = [data.embedding for data in response.data]
            
            logger.info(f"Generated {len(embeddings)} embeddings using {self.model}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _get_cache_path(self, text_hash: str) -> Path:
        """
        Get cache file path for a text hash.
        
        Args:
            text_hash: Hash of the text
            
        Returns:
            Path to the cache file
        """
        if self.cache_dir is None:
            raise ValueError("Cache directory not set")
            
        return self.cache_dir / f"{text_hash}_{self.model}.json"
    
    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """
        Save an embedding to the cache.
        
        Args:
            text: Original text
            embedding: Embedding vector
        """
        if not self.use_cache or self.cache_dir is None:
            return
            
        # Generate hash for the text
        text_hash = get_text_hash(text)
        
        # Get cache file path
        cache_path = self._get_cache_path(text_hash)
        
        # Save embedding to cache
        cache_data = {
            "text_hash": text_hash,
            "model": self.model,
            "embedding": embedding,
            "timestamp": time.time(),
        }
        
        save_json(cache_data, cache_path)
        logger.debug(f"Cached embedding for text hash {text_hash}")
    
    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Get an embedding from the cache.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding vector if found in cache, None otherwise
        """
        if not self.use_cache or self.cache_dir is None:
            return None
            
        # Generate hash for the text
        text_hash = get_text_hash(text)
        
        # Get cache file path
        cache_path = self._get_cache_path(text_hash)
        
        # Check if cache file exists
        if not cache_path.exists():
            return None
            
        try:
            # Load embedding from cache
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
                
            # Check if the cached embedding is for the right model
            if cache_data.get("model") != self.model:
                logger.warning(f"Cached embedding is for model {cache_data.get('model')}, not {self.model}")
                return None
                
            logger.debug(f"Using cached embedding for text hash {text_hash}")
            return cache_data.get("embedding")
            
        except Exception as e:
            logger.warning(f"Error loading embedding from cache: {e}")
            return None
    
    def get_embeddings(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to embed in a single API call
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        embeddings = []
        texts_to_embed = []
        texts_to_embed_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached_embedding = self._get_from_cache(text)
            
            if cached_embedding is not None:
                # Use cached embedding
                while len(embeddings) <= i:
                    embeddings.append(None)
                embeddings[i] = cached_embedding
            else:
                # Need to get embedding from API
                texts_to_embed.append(text)
                texts_to_embed_indices.append(i)
                
        # Get embeddings from API in batches
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i+batch_size]
            
            # Get embeddings from API
            batch_embeddings = self._get_embeddings_from_api(batch_texts)
            
            # Save embeddings to cache and insert into result list
            for j, embedding in enumerate(batch_embeddings):
                text_idx = texts_to_embed_indices[i + j]
                text = texts_to_embed[i + j]
                
                # Save to cache
                self._save_to_cache(text, embedding)
                
                # Insert into result list
                while len(embeddings) <= text_idx:
                    embeddings.append(None)
                embeddings[text_idx] = embedding
                
        return embeddings
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.get_embeddings([text])
        return embeddings[0]
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, List[float]]:
        """
        Generate embeddings for a list of document chunks.
        
        Args:
            chunks: List of document chunks to embed
            
        Returns:
            Dictionary mapping chunk IDs to embeddings
        """
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Get embeddings
        embeddings = self.get_embeddings(texts)
        
        # Map chunk IDs to embeddings
        return dict(zip(chunk_ids, embeddings))
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.get_embedding(query) 