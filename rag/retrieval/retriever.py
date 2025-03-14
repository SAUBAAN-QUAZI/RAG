"""
Retriever Module
------------
This module coordinates embedding and search to retrieve relevant documents.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Union, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

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
import uuid


class CrossEncoder:
    """
    A basic cross-encoder implementation for reranking results.
    
    This class uses a pre-trained embedding model to compute similarities
    between pairs of texts (query and result) for more accurate ranking.
    """
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize the CrossEncoder.
        
        Args:
            embedding_service: Service for generating embeddings
        """
        # Create embedding service if not provided
        if embedding_service is None:
            self.embedding_service = EmbeddingService()
        else:
            self.embedding_service = embedding_service
            
        logger.info("Initialized CrossEncoder for reranking")
        
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The query string
            documents: List of document dictionaries
            top_k: Number of top results to return
            
        Returns:
            List of reranked document dictionaries
        """
        if not documents:
            return []
            
        logger.info(f"Reranking {len(documents)} documents using cross-encoder")
        
        # For each document, create a combined embedding of query and content pair
        doc_scores = []
        
        for doc in documents:
            content = doc.get("content", "")
            # Compute cross-embedding similarity
            try:
                # Get embeddings for query and document
                query_embedding = self.embedding_service.embed_query(query)
                doc_embedding = self.embedding_service.embed_text(content)
                
                # Compute cosine similarity
                similarity = self._compute_cosine_similarity(query_embedding, doc_embedding)
                
                # Store original similarity for debugging
                original_similarity = doc.get("similarity", 0)
                
                # Calculate a weighted score: 70% cross-encoder, 30% original
                combined_score = 0.7 * similarity + 0.3 * original_similarity
                
                doc_scores.append((doc, combined_score))
                
            except Exception as e:
                logger.error(f"Error computing cross-encoder score: {str(e)}")
                # Fall back to original score if there was an error
                doc_scores.append((doc, doc.get("similarity", 0)))
        
        # Sort by score in descending order
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k results and update scores
        reranked_docs = []
        for doc, score in doc_scores[:top_k]:
            # Update similarity score
            reranked_doc = doc.copy()
            reranked_doc.update({"similarity": score, "reranked": True})
            reranked_docs.append(reranked_doc)
            
        logger.info(f"Reranked documents, returning top {len(reranked_docs)} results")
        return reranked_docs
    
    def _compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Handle division by zero
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (norm1 * norm2)


class KeywordSearcher:
    """
    Performs keyword-based search using TF-IDF.
    """
    
    def __init__(self):
        """
        Initialize the KeywordSearcher.
        """
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2),  # Include unigrams and bigrams
            max_df=0.85,  # Ignore terms that appear in more than 85% of documents
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            stop_words='english'
        )
        self.doc_vectors = None
        self.documents = []
        
        logger.info("Initialized KeywordSearcher")
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the keyword searcher.
        
        Args:
            documents: List of document dictionaries with 'content' field
        """
        if not documents:
            logger.warning("No documents to add to keyword searcher")
            return
            
        logger.info(f"Adding {len(documents)} documents to keyword searcher")
        
        # Extract content from documents
        contents = [doc.get("content", "") for doc in documents]
        
        # Fit vectorizer and transform documents
        try:
            self.doc_vectors = self.vectorizer.fit_transform(contents)
            self.documents = documents
            logger.info(f"Successfully added {len(documents)} documents to keyword searcher")
        except Exception as e:
            logger.error(f"Error adding documents to keyword searcher: {str(e)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant documents using keywords.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        if not self.documents or self.doc_vectors is None:
            logger.warning("No documents in keyword searcher")
            return []
            
        logger.info(f"Performing keyword search for query: {query}")
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Compute similarities
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Get top_k results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Create result dictionaries
            results = []
            for i in top_indices:
                if similarities[i] > 0:  # Only include results with some similarity
                    doc_copy = self.documents[i].copy()
                    doc_copy.update({"similarity": float(similarities[i]), "source": "keyword"})
                    results.append(doc_copy)
            
            logger.info(f"Found {len(results)} results with keyword search")
            return results
            
        except Exception as e:
            logger.error(f"Error performing keyword search: {str(e)}")
            return []


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
        enable_hybrid_search: bool = True,
        enable_reranking: bool = True,
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
            enable_hybrid_search: Whether to enable hybrid search (vector + keyword)
            enable_reranking: Whether to enable reranking of results
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
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_reranking = enable_reranking
        
        # Initialize components for advanced retrieval
        if self.enable_hybrid_search:
            self.keyword_searcher = KeywordSearcher()
        else:
            self.keyword_searcher = None
            
        if self.enable_reranking:
            self.reranker = CrossEncoder(embedding_service=self.embedding_service)
        else:
            self.reranker = None
        
        # Keep track of all chunks for keyword search
        self.all_chunks = []
        
        logger.info(f"Initialized Retriever with top_k={top_k}, "
                   f"similarity_threshold={similarity_threshold}, "
                   f"hybrid_search={enable_hybrid_search}, "
                   f"reranking={enable_reranking}")
    
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
        
        # Generate embeddings
        chunk_embeddings = self.embedding_service.embed_chunks(chunks)
        
        # Debug mode: dump information about chunks and embeddings
        if os.environ.get("RAG_DEBUG", "").lower() in ("1", "true", "yes"):
            from rag.utils.utils import debug_chunk_embeddings
            debug_dir = Path("data/debug") / f"retriever_debug_{int(time.time())}"
            debug_chunk_embeddings(chunks, chunk_embeddings, debug_dir)
        
        # Add embeddings to vector store
        try:
            logger.debug(f"Adding embeddings to vector store: {len(chunk_embeddings)} embeddings")
            self.vector_store.add_embeddings(chunk_embeddings, chunks)
        except Exception as e:
            logger.error(f"Error adding embeddings to vector store: {str(e)}")
            
        # Add chunks to keyword searcher for hybrid search
        if self.enable_hybrid_search:
            try:
                # Prepare documents for keyword search
                documents = []
                for chunk in chunks:
                    doc = {
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "metadata": chunk.metadata
                    }
                    documents.append(doc)
                    
                self.keyword_searcher.add_documents(documents)
                logger.info(f"Successfully added {len(documents)} documents to keyword searcher")
            except Exception as e:
                logger.error(f"Error adding documents to keyword searcher: {str(e)}")
                
        logger.info(f"Added {len(chunks)} chunks to retrieval system")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant document chunks for a query using hybrid search and reranking.
        
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
        
        # Step 1: Perform vector search
        vector_results = self._vector_search(query, top_k * 2, filter_dict)  # Get more results for reranking
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        # Step 2: Perform keyword search if enabled
        keyword_results = []
        if self.enable_hybrid_search and self.keyword_searcher is not None:
            keyword_results = self.keyword_searcher.search(query, top_k=top_k)
            logger.info(f"Keyword search returned {len(keyword_results)} results")
        
        # Step 3: Merge results
        merged_results = self._merge_results(vector_results, keyword_results)
        logger.info(f"Merged results: {len(merged_results)} documents")
        
        # Step 4: Apply reranking if enabled
        if self.enable_reranking and self.reranker is not None:
            results = self.reranker.rerank(query, merged_results, top_k=top_k)
        else:
            # Just sort by similarity and take top_k
            results = sorted(merged_results, key=lambda x: x.get("similarity", 0), reverse=True)[:top_k]
        
        logger.info(f"Retrieved {len(results)} documents for query")
        return results
    
    def _vector_search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Perform vector similarity search.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List of dictionaries containing retrieved chunks and similarity scores
        """
        # Generate embedding for query
        query_embedding = self.embedding_service.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
        )
        
        # Add source information
        for result in results:
            result["source"] = "vector"
            
        return results
    
    def _merge_results(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
    ) -> List[Dict]:
        """
        Merge vector and keyword search results with deduplication.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            
        Returns:
            Merged and deduplicated results with updated scores
        """
        if not keyword_results:
            return vector_results
        
        # Create ID mapping for vector results for deduplication
        vector_ids = {r.get("chunk_id", i): i for i, r in enumerate(vector_results)}
        
        # Merge results with deduplication
        merged_results = vector_results.copy()
        
        for kw_result in keyword_results:
            chunk_id = kw_result.get("chunk_id")
            
            if chunk_id in vector_ids:
                # Document exists in vector results, update score
                vector_idx = vector_ids[chunk_id]
                vector_score = vector_results[vector_idx].get("similarity", 0)
                keyword_score = kw_result.get("similarity", 0)
                
                # Combine scores: 70% vector, 30% keyword
                combined_score = 0.7 * vector_score + 0.3 * keyword_score
                merged_results[vector_idx]["similarity"] = combined_score
                merged_results[vector_idx]["source"] = "hybrid"
            else:
                # New document, add to results
                merged_results.append(kw_result)
        
        # Sort by similarity
        merged_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        return merged_results
    
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
            source = result.get("source", "unknown")
            content_preview = result.get("content", "")[:50] if result.get("content") else "No content"
            logger.info(f"Result {i+1}: chunk_id={chunk_id}, similarity={similarity:.4f}, source={source}, preview={content_preview}...")
        
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
            source = result.get("source", "vector")
            
            # Include relevant metadata in context
            doc_source = metadata.get("source", "Unknown")
            if "page_count" in metadata and "chunk_index" in metadata:
                page_info = f"Page {metadata.get('chunk_index', 0) + 1} of {metadata.get('page_count', 1)}"
            else:
                page_info = ""
                
            # Format chunk with metadata
            formatted_chunk = (
                f"[Document {i+1}] {doc_source} {page_info}\n"
                f"{content}\n"
                f"Relevance: {similarity:.2f} (via {source})\n"
            )
            
            formatted_chunks.append(formatted_chunk)
            
        # Combine chunks into a single context string
        context = "\n\n".join(formatted_chunks)
        logger.info(f"Created context with {len(formatted_chunks)} chunks, total length: {len(context)} characters")
        
        return context 