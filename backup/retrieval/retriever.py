"""
Retriever Module
------------
This module coordinates embedding and search to retrieve relevant documents.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Union, Tuple, Any
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
        vector_store_type: Optional[str] = None,  # Kept for backward compatibility
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
            vector_store_type: Type of vector store to use (only 'qdrant' is supported)
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
    
    def _analyze_query_and_adjust_params(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query and adjust retrieval parameters accordingly.
        
        This method examines query characteristics such as length, complexity, and intent
        to optimize retrieval parameters for better results. It handles different types
        of queries differently:
        
        1. Short factual queries: More results with higher threshold
        2. Long complex queries: Fewer, more targeted results
        3. Section-specific queries: Special handling for chapter and section references
        4. Technical queries: Balanced vector and keyword weights
        
        Args:
            query: The user's query string
            
        Returns:
            Dictionary of adjusted parameters
        """
        # Clean the query
        clean_query = query.strip().lower()
        
        # Default parameters
        params = {
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "vector_weight": 0.5,  # Balance between vector and keyword search
            "keyword_weight": 0.5,
            "enable_reranking": self.enable_reranking,
        }
        
        # Check query length (word count)
        words = clean_query.split()
        word_count = len(words)
        
        # Track query characteristics
        is_long_query = word_count > 15
        is_short_query = word_count < 5
        
        # Check for presence of technical terms using a simple heuristic
        technical_terms = ["algorithm", "function", "method", "api", "framework", 
                           "library", "implementation", "architecture", "model",
                           "system", "protocol", "interface", "component", "module"]
        
        has_technical_terms = any(term in clean_query for term in technical_terms)
        
        # Detect if query is about finding a specific section
        section_patterns = [
            r"chapter\s+(\d+|[ivxlcdm]+)",  # chapter 1, chapter iv
            r"section\s+(\d+(\.\d+)*)",     # section 1, section 1.2
            r"part\s+(\d+|[ivxlcdm]+)",     # part 1, part ii
            r"page\s+(\d+)",                # page 42
            r"paragraph\s+(\d+)",           # paragraph 3
            r"appendix\s+([a-z]|\d+)",      # appendix a, appendix 1
            r"figure\s+(\d+(\.\d+)*)",      # figure 1, figure 2.3
            r"table\s+(\d+(\.\d+)*)",       # table 1, table 4.2
            r"in\s+chapter\s+(\d+|[ivxlcdm]+)",  # in chapter 1
            r"from\s+section\s+(\d+(\.\d+)*)",   # from section 1.2
        ]
        
        # Extract structural references with their positions
        structural_references = []
        for pattern in section_patterns:
            matches = re.finditer(pattern, clean_query, re.IGNORECASE)
            for match in matches:
                ref_type = match.group(0).split()[0].lower()  # chapter, section, etc.
                ref_value = match.group(1)                    # the number/identifier
                structural_references.append({
                    "type": ref_type,
                    "value": ref_value,
                    "span": match.span(),
                    "original": match.group(0)
                })
        
        # Sort references by their position in the query
        structural_references.sort(key=lambda x: x["span"][0])
        
        # If we have structural references, adjust parameters for targeted retrieval
        if structural_references:
            # Get the first reference (typically most important)
            primary_ref = structural_references[0]
            
            # Create a more targeted search query that focuses on finding the specific section
            # Keep the original query but boost the reference
            logger.info(f"Query contains reference to {primary_ref['type']} {primary_ref['value']}")
            
            # For explicit section requests, prioritize keyword search more
            params["top_k"] = max(self.top_k + 4, 8)  # Increase result count
            params["similarity_threshold"] = 0.3  # Lower threshold to get more matches
            params["vector_weight"] = 0.3  # Reduce vector search weight
            params["keyword_weight"] = 0.7  # Increase keyword weight
            params["enable_reranking"] = True  # Always rerank for section queries
            
            # For chapter/section queries, add special keyword boosting
            if "explain" in clean_query and any(ref["type"] == "chapter" for ref in structural_references):
                # This is likely a query asking to explain a chapter
                chapter_ref = next(ref for ref in structural_references if ref["type"] == "chapter")
                # Structure-aware retrieval adjustments
                params["top_k"] = max(self.top_k + 6, 10)  # Get more results for chapters
                params["similarity_threshold"] = 0.25  # Lower threshold substantially
            
            return params
            
        # For long queries, prefer more targeted retrieval
        if is_long_query:
            params["top_k"] = max(3, self.top_k - 2)  # Reduce result count
            params["similarity_threshold"] = min(0.7, self.similarity_threshold + 0.1)  # Increase threshold
            params["vector_weight"] = 0.7  # Prioritize vector search for semantic understanding
            params["keyword_weight"] = 0.3
        
        # For short queries, get more results with lower threshold
        elif is_short_query:
            params["top_k"] = min(12, self.top_k + 4)  # Increase result count
            params["similarity_threshold"] = max(0.4, self.similarity_threshold - 0.1)  # Decrease threshold
            # Balance between vector and keyword for short queries
            params["vector_weight"] = 0.5
            params["keyword_weight"] = 0.5
        
        # For technical queries, use a balanced approach
        if has_technical_terms:
            params["vector_weight"] = 0.6  # Slightly favor vector search
            params["keyword_weight"] = 0.4
            params["enable_reranking"] = True  # Enable reranking for technical queries
        
        # For typical knowledge-seeking queries, favor vector search
        if any(w in query.lower() for w in ["what", "how", "explain", "describe", "why"]):
            params["vector_weight"] = 0.6
            params["keyword_weight"] = 0.4
        
        # For specific information lookup, favor keyword search
        if any(w in query.lower() for w in ["where", "when", "who", "which", "find"]):
            params["vector_weight"] = 0.4
            params["keyword_weight"] = 0.6
        
        logger.info(f"Adjusted retrieval parameters for query: {params}")
        return params

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query with dynamic parameter adjustment.
        
        This enhanced retrieval method:
        1. Analyzes the query to dynamically adjust retrieval parameters
        2. Performs hybrid search combining vector and keyword approaches
        3. Applies document coherence boosting in result ranking
        4. Optionally reranks results with cross-encoder for improved relevance
        
        Args:
            query: Query text
            top_k: Maximum number of results to return (overrides dynamic adjustment if provided)
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List of retrieved documents with similarity scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to retriever")
            return []
            
        # Dynamically adjust retrieval parameters based on query
        params = self._analyze_query_and_adjust_params(query)
        
        # Override with explicit top_k if provided
        if top_k is not None:
            params["top_k"] = top_k
        
        # Get search results using adjusted parameters
        search_limit = max(params["top_k"] * 3, 10)  # Get more results for reranking
        
        # Vector search
        logger.info(f"Searching vector store with top_k={params['top_k']}, search_limit={search_limit}, filters={filter_dict}")
        vector_results = self._vector_search(query, top_k=search_limit, filter_dict=filter_dict)
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        # Keyword search if hybrid search is enabled
        keyword_results = []
        if self.enable_hybrid_search:
            logger.info(f"Performing keyword search for query: {query}")
            keyword_results = self.keyword_searcher.search(query, top_k=search_limit)
            logger.info(f"Keyword search returned {len(keyword_results)} results")
        
        # Merge results with enhanced merging that considers document coherence
        merged_results = self._merge_results(vector_results, keyword_results)
        
        # Apply custom weights for vector and keyword results
        for result in merged_results:
            if "source" in result:
                if result["source"] == "vector":
                    result["similarity"] = result["similarity"] * params["vector_weight"]
                elif result["source"] == "keyword":
                    result["similarity"] = result["similarity"] * params["keyword_weight"]
        
        # Re-sort after applying weights
        merged_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Filter by similarity threshold
        threshold_results = [
            r for r in merged_results 
            if r.get("similarity", 0) >= params["similarity_threshold"]
        ]
        
        # If we don't have enough results after threshold filtering, use at least a minimum number
        min_results = min(3, len(merged_results))
        if len(threshold_results) < min_results:
            threshold_results = merged_results[:min_results]
        
        # Apply cross-encoder reranking if enabled
        final_results = threshold_results
        if params["enable_reranking"] and self.reranker and len(threshold_results) > 1:
            logger.info(f"Reranking {len(threshold_results)} documents using cross-encoder")
            final_results = self.reranker.rerank(query, threshold_results, top_k=params["top_k"])
            logger.info(f"Reranked documents, returning top {len(final_results)} results")
        else:
            # Limit to top_k without reranking
            final_results = threshold_results[:params["top_k"]]
        
        return final_results
    
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
        Merge vector search and keyword search results with intelligent weighting.
        
        This enhanced merging algorithm:
        1. Considers both relevance scores and result positions
        2. Boosts results that appear in both vector and keyword searches
        3. Groups related document chunks for more coherent context
        4. Considers document structure for better result organization
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            
        Returns:
            Merged and reranked search results
        """
        if not vector_results and not keyword_results:
            return []
            
        # Create dictionaries for faster lookup
        vector_dict = {r["chunk_id"]: r for r in vector_results}
        keyword_dict = {r["chunk_id"]: r for r in keyword_results}
        
        # Find document IDs for all results to enable structural grouping
        all_chunks = {}  # Maps chunk_id to result
        doc_to_chunks = {}  # Maps doc_id to a list of its chunks
        
        # Process vector results
        for i, result in enumerate(vector_results):
            chunk_id = result["chunk_id"]
            doc_id = result.get("metadata", {}).get("doc_id", "unknown")
            
            # Add position info to help with ranking
            result["vector_rank"] = i + 1
            
            # Store in mappings
            all_chunks[chunk_id] = result
            
            # Group by document
            if doc_id not in doc_to_chunks:
                doc_to_chunks[doc_id] = []
            doc_to_chunks[doc_id].append(chunk_id)
            
        # Process keyword results
        for i, result in enumerate(keyword_results):
            chunk_id = result["chunk_id"]
            doc_id = result.get("metadata", {}).get("doc_id", "unknown")
            
            # Add position info
            result["keyword_rank"] = i + 1
            
            # Update existing result or add new one
            if chunk_id in all_chunks:
                # Update existing chunk with keyword info
                all_chunks[chunk_id]["keyword_rank"] = i + 1
                # Average the scores for chunks that appear in both results
                vector_score = all_chunks[chunk_id].get("similarity", 0)
                keyword_score = result.get("similarity", 0)
                # Boost scores for results in both lists
                all_chunks[chunk_id]["similarity"] = (vector_score + keyword_score) * 1.2
            else:
                # Add new chunk from keyword results
                all_chunks[chunk_id] = result
                
                # Group by document
                if doc_id not in doc_to_chunks:
                    doc_to_chunks[doc_id] = []
                doc_to_chunks[doc_id].append(chunk_id)
        
        # Calculate final scores with a weighted approach
        merged_results = []
        for chunk_id, result in all_chunks.items():
            # Default ranks if not present
            vector_rank = result.get("vector_rank", len(vector_results) + 1 if vector_results else 1)
            keyword_rank = result.get("keyword_rank", len(keyword_results) + 1 if keyword_results else 1)
            
            # Normalize ranks to [0, 1] range where 1 is best
            norm_vector_rank = 1.0 - min(1.0, vector_rank / (len(vector_results) + 1)) if vector_results else 0
            norm_keyword_rank = 1.0 - min(1.0, keyword_rank / (len(keyword_results) + 1)) if keyword_results else 0
            
            # Weight vector results more heavily but consider both
            weighted_score = (norm_vector_rank * 0.7) + (norm_keyword_rank * 0.3)
            
            # Apply boost for results in both searches
            if "vector_rank" in result and "keyword_rank" in result:
                weighted_score *= 1.2
                
            # Find neighboring chunks and apply coherence boost
            doc_id = result.get("metadata", {}).get("doc_id", "unknown")
            doc_chunks = doc_to_chunks.get(doc_id, [])
            
            # Calculate position within document (if available)
            chunk_index = -1
            if "metadata" in result and "chunk_index" in result["metadata"]:
                chunk_index = result["metadata"]["chunk_index"]
                
            # Check for chunks adjacent to high-scoring ones
            for other_id in doc_chunks:
                if other_id == chunk_id:
                    continue
                    
                other_result = all_chunks[other_id]
                other_index = other_result.get("metadata", {}).get("chunk_index", -1)
                
                # If the chunks are adjacent in the document, boost scoring
                if chunk_index >= 0 and other_index >= 0 and abs(chunk_index - other_index) == 1:
                    # Apply a smaller boost for adjacency
                    weighted_score *= 1.05
            
            # Store the final score
            result["final_score"] = weighted_score
            merged_results.append(result)
            
        # Sort by final score
        merged_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        # Remove temporary ranking fields
        for result in merged_results:
            if "vector_rank" in result:
                del result["vector_rank"]
            if "keyword_rank" in result:
                del result["keyword_rank"]
            if "final_score" in result:
                # Rename to similarity for consistency with original implementation
                result["similarity"] = result["final_score"]
                del result["final_score"]
        
        return merged_results
    
    def get_relevant_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
    ) -> str:
        """
        Get a formatted context string from the most relevant documents.
        
        This method retrieves documents based on the query, then formats them
        into a single context string for use in augmented generation.
        
        Args:
            query: The user query
            top_k: Number of results to retrieve (overrides default)
            filter_dict: Optional metadata filters
            
        Returns:
            String containing formatted relevant context
        """
        # Analyze the query to detect if it's about a specific chapter or section
        clean_query = query.strip().lower()
        is_chapter_query = re.search(r"chapter\s+(\d+|[ivxlcdm]+)", clean_query, re.IGNORECASE) is not None
        
        # Retrieve relevant documents
        results = self.retrieve(query, top_k, filter_dict)
        
        # Handle the case where no results are found
        if not results:
            return "No relevant information found."
        
        # Special handling for chapter queries - attempt to order by document position
        if is_chapter_query:
            # Try to extract the chapter number from the query
            chapter_match = re.search(r"chapter\s+(\d+|[ivxlcdm]+)", clean_query, re.IGNORECASE)
            if chapter_match:
                chapter_number = chapter_match.group(1)
                logger.info(f"Creating context for chapter {chapter_number} query")
                
                # Sort results if they have page numbers to maintain document order
                has_page_numbers = all('page_number' in result.get('metadata', {}) for result in results)
                if has_page_numbers:
                    results = sorted(results, key=lambda x: x.get('metadata', {}).get('page_number', 0))
        
        # Format the context
        context_parts = []
        
        for i, result in enumerate(results):
            # Format metadata for the context
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page_info = f"page {metadata.get('page_number', 'unknown')}" if "page_number" in metadata else ""
            
            # Clean up source path for display
            if isinstance(source, str):
                source = os.path.basename(source)
            
            # Format the header for each context chunk
            header = f"Document {i+1}: {source}"
            if page_info:
                header += f" ({page_info})"
            
            # Add document section information if available
            if "section" in metadata:
                header += f" - {metadata['section']}"
            elif "chapter" in metadata:
                header += f" - Chapter {metadata['chapter']}"
                
            # Format the content
            content = result.get("content", "").strip()
            
            # Add to context parts
            context_parts.append(f"{header}\n{content}")
        
        # Join all parts with clear separators
        context = "\n\n" + "\n\n---\n\n".join(context_parts)
        
        # Log context creation
        logger.info(f"Created context with {len(results)} chunks, total length: {len(context)} characters ")
        
        return context 