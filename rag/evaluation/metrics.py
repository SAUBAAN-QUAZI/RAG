"""
Evaluation Metrics
----------------
This module contains metrics for evaluating retrieval and generation performance.
"""

import math
import numpy as np
from typing import List, Dict, Any, Set, Union, Optional
from scipy.spatial.distance import cosine
from collections import Counter

def calculate_precision(retrieved_ids: Set[str], relevant_ids: Set[str]) -> float:
    """
    Calculate precision (proportion of retrieved documents that are relevant).
    
    Args:
        retrieved_ids: Set of retrieved document/chunk IDs
        relevant_ids: Set of relevant document/chunk IDs (ground truth)
        
    Returns:
        float: Precision score (0.0 to 1.0)
    """
    if not retrieved_ids:
        return 0.0
    
    intersection = retrieved_ids.intersection(relevant_ids)
    return len(intersection) / len(retrieved_ids)

def calculate_recall(retrieved_ids: Set[str], relevant_ids: Set[str]) -> float:
    """
    Calculate recall (proportion of relevant documents that are retrieved).
    
    Args:
        retrieved_ids: Set of retrieved document/chunk IDs
        relevant_ids: Set of relevant document/chunk IDs (ground truth)
        
    Returns:
        float: Recall score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 1.0  # All relevant documents were retrieved (none exist)
    
    intersection = retrieved_ids.intersection(relevant_ids)
    return len(intersection) / len(relevant_ids)

def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1 score (harmonic mean of precision and recall).
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        float: F1 score (0.0 to 1.0)
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def calculate_mrr(results: List[Dict[str, Any]], relevant_ids: Set[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        results: List of retrieval results with 'chunk_id' key
        relevant_ids: Set of relevant document/chunk IDs (ground truth)
        
    Returns:
        float: MRR score (0.0 to 1.0)
    """
    for i, result in enumerate(results):
        if result["chunk_id"] in relevant_ids:
            return 1.0 / (i + 1)
    
    return 0.0

def calculate_ndcg(results: List[Dict[str, Any]], relevant_ids: Set[str], 
                   relevance_scores: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).
    
    Args:
        results: List of retrieval results with 'chunk_id' key
        relevant_ids: Set of relevant document/chunk IDs (ground truth)
        relevance_scores: Optional dictionary mapping chunk_ids to relevance scores
        
    Returns:
        float: NDCG score (0.0 to 1.0)
    """
    if not results or not relevant_ids:
        return 0.0
    
    # If no relevance scores provided, use binary relevance
    if relevance_scores is None:
        relevance_scores = {chunk_id: 1.0 for chunk_id in relevant_ids}
        
    # Calculate DCG
    dcg = 0.0
    for i, result in enumerate(results):
        if result["chunk_id"] in relevant_ids:
            rel_score = relevance_scores.get(result["chunk_id"], 1.0)
            dcg += rel_score / math.log2(i + 2)  # +2 because log2(1) = 0
            
    # Calculate ideal DCG (best possible ranking)
    ideal_ordering = sorted(relevant_ids, 
                          key=lambda chunk_id: relevance_scores.get(chunk_id, 1.0),
                          reverse=True)
    idcg = 0.0
    for i, chunk_id in enumerate(ideal_ordering[:len(results)]):
        rel_score = relevance_scores.get(chunk_id, 1.0)
        idcg += rel_score / math.log2(i + 2)
        
    if idcg == 0:
        return 0.0
        
    return dcg / idcg

def calculate_semantic_similarity(text1: str, text2: str, embedding_service=None) -> float:
    """
    Calculate semantic similarity between two texts using embeddings.
    
    Args:
        text1: First text
        text2: Second text
        embedding_service: Optional embedding service instance
        
    Returns:
        float: Similarity score (0.0 to 1.0)
    """
    if not embedding_service:
        # Lazy import to avoid circular dependencies
        from rag.embedding.service import EmbeddingService
        embedding_service = EmbeddingService()
    
    # Get embeddings
    embedding1 = embedding_service.get_embedding(text1)
    embedding2 = embedding_service.get_embedding(text2)
    
    # Calculate cosine similarity (1 - cosine distance)
    similarity = 1 - cosine(embedding1, embedding2)
    return float(similarity)

def calculate_ngram_overlap(generated: str, reference: str, n: int = 1) -> float:
    """
    Calculate n-gram overlap between generated text and reference.
    
    Args:
        generated: Generated text
        reference: Reference text
        n: Size of n-grams
        
    Returns:
        float: Overlap score (0.0 to 1.0)
    """
    def get_ngrams(text, n):
        tokens = text.lower().split()
        return set(' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    gen_ngrams = get_ngrams(generated, n)
    ref_ngrams = get_ngrams(reference, n)
    
    if not ref_ngrams:
        return 0.0
    
    overlap = gen_ngrams.intersection(ref_ngrams)
    return len(overlap) / len(ref_ngrams)

def calculate_factual_consistency(generated: str, context: str, threshold: float = 0.5) -> float:
    """
    Calculate factual consistency between generated text and source context.
    
    This is a simplified implementation using n-gram overlap.
    For production use, consider using a trained model specifically for factual consistency.
    
    Args:
        generated: Generated text
        context: Source context
        threshold: Minimum overlap threshold to consider a fact supported
        
    Returns:
        float: Factual consistency score (0.0 to 1.0)
    """
    # Split into sentences (simplified)
    gen_sentences = [s.strip() for s in generated.split('.') if s.strip()]
    
    # Calculate overlap for each generated sentence
    consistency_scores = []
    for sentence in gen_sentences:
        # Skip very short sentences as they might be generic
        if len(sentence.split()) < 4:
            continue
            
        # Calculate overlap with context (using both unigrams and bigrams)
        unigram_overlap = calculate_ngram_overlap(sentence, context, n=1)
        bigram_overlap = calculate_ngram_overlap(sentence, context, n=2)
        
        # Combine scores with more weight to bigrams
        sentence_score = (unigram_overlap + 2 * bigram_overlap) / 3
        consistency_scores.append(sentence_score >= threshold)
    
    if not consistency_scores:
        return 0.0
        
    # Return proportion of consistent sentences
    return sum(consistency_scores) / len(consistency_scores)

def calculate_answer_relevance(generated: str, question: str, embedding_service=None) -> float:
    """
    Calculate relevance of the answer to the question.
    
    Args:
        generated: Generated answer text
        question: Question text
        embedding_service: Optional embedding service instance
        
    Returns:
        float: Relevance score (0.0 to 1.0)
    """
    return calculate_semantic_similarity(generated, question, embedding_service) 