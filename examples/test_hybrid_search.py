#!/usr/bin/env python
"""
Testing Hybrid Search Performance
--------------------------------
This script evaluates the performance of the hybrid search feature 
(combining vector search with keyword-based search) compared to 
vector-only search.
"""

import sys
import time
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

# Add project root to path if running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.document_processing import process_document
from rag.document_processing.document import DocumentChunk
from rag.embedding.service import EmbeddingService
from rag.retrieval.retriever import Retriever, KeywordSearcher
from rag.utils import logger, save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Path to the test document
DOCUMENT_PATH = "data/documents/LLM book.pdf"

# Path for evaluation results
EVAL_DIR = Path("data/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Test queries designed to highlight different retrieval strengths
TEST_QUERIES = [
    # Queries that work better with vector search (semantic similarity)
    "Explain neural architecture of transformers",  # Semantic: "transformer architecture"
    "How does attention help with understanding language?",  # Semantic: "attention mechanism"
    "Limitations of language models for reasoning",  # Semantic: "LLM limitations"
    
    # Queries that work better with keyword search (exact matches)
    "What is RLHF?",  # Keyword: "RLHF"
    "Explain the GPT-4 architecture",  # Keyword: "GPT-4"
    "What is the impact of context window size?",  # Keyword: "context window"
]

def process_and_prepare_data(file_path: str) -> List[DocumentChunk]:
    """
    Process document and prepare data for retrieval testing.
    
    Args:
        file_path: Path to document file
    
    Returns:
        List of document chunks
    """
    logger.info(f"Processing document: {file_path}")
    
    # Process document with large chunks and semantic splitting for better quality
    result = process_document(
        file_path=file_path,
        splitter_type="semantic",
        chunk_size=1500,
        chunk_overlap=300,
    )
    
    document = result["document"]
    chunks = result["chunks"]
    
    logger.info(f"Document {document.doc_id} processed into {len(chunks)} chunks")
    
    # Save a preview of chunks for inspection
    chunks_preview = [
        {
            "chunk_id": chunk.metadata.get("chunk_id", "unknown"),
            "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            "token_count": len(chunk.content.split()),
        }
        for chunk in chunks[:5]  # Just save a few for preview
    ]
    
    save_json(
        data=chunks_preview,
        file_path=EVAL_DIR / "hybrid_search_chunks_preview.json"
    )
    
    return chunks

def test_retrieval_methods(
    chunks: List[DocumentChunk], 
    queries: List[str], 
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Test different retrieval methods and compare results.
    
    Args:
        chunks: List of document chunks
        queries: List of test queries
        top_k: Number of results to retrieve
        
    Returns:
        Dictionary of test results
    """
    logger.info("Setting up retrievers for testing")
    
    # Create embedding service
    embedding_service = EmbeddingService()
    
    # Initialize retrievers with different configurations
    vector_only_retriever = Retriever(
        embedding_service=embedding_service,
        collection_name="test_vector_only",
        enable_hybrid_search=False,
        enable_reranking=False,
    )
    
    hybrid_retriever = Retriever(
        embedding_service=embedding_service,
        collection_name="test_hybrid",
        enable_hybrid_search=True,
        enable_reranking=False,
    )
    
    hybrid_with_rerank_retriever = Retriever(
        embedding_service=embedding_service,
        collection_name="test_hybrid_rerank",
        enable_hybrid_search=True,
        enable_reranking=True,
    )
    
    # Add chunks to retrievers
    logger.info("Adding chunks to retrievers")
    vector_only_retriever.add_chunks(chunks)
    hybrid_retriever.add_chunks(chunks)
    hybrid_with_rerank_retriever.add_chunks(chunks)
    
    # Initialize results
    results = {
        "vector_only": [],
        "hybrid": [],
        "hybrid_with_rerank": [],
        "query_times": {},
        "summary": {}
    }
    
    # Test each query
    for i, query in enumerate(queries):
        logger.info(f"Testing query {i+1}/{len(queries)}: {query}")
        query_results = {
            "query": query,
            "vector_only": {},
            "hybrid": {},
            "hybrid_with_rerank": {}
        }
        
        # Test vector-only retriever
        start_time = time.time()
        vector_results = vector_only_retriever.retrieve(query=query, top_k=top_k)
        vector_time = time.time() - start_time
        
        # Test hybrid retriever
        start_time = time.time()
        hybrid_results = hybrid_retriever.retrieve(query=query, top_k=top_k)
        hybrid_time = time.time() - start_time
        
        # Test hybrid retriever with reranking
        start_time = time.time()
        hybrid_rerank_results = hybrid_with_rerank_retriever.retrieve(query=query, top_k=top_k)
        hybrid_rerank_time = time.time() - start_time
        
        # Record timings
        query_results["vector_only"]["time"] = vector_time
        query_results["hybrid"]["time"] = hybrid_time
        query_results["hybrid_with_rerank"]["time"] = hybrid_rerank_time
        
        # Record results
        query_results["vector_only"]["results"] = format_results(vector_results)
        query_results["hybrid"]["results"] = format_results(hybrid_results)
        query_results["hybrid_with_rerank"]["results"] = format_results(hybrid_rerank_results)
        
        # Compare result sets
        query_results["result_overlap"] = calculate_result_overlap(
            vector_results, hybrid_results, hybrid_rerank_results
        )
        
        # Add to overall results
        results["vector_only"].append({
            "query": query,
            "time": vector_time,
            "result_count": len(vector_results),
            "results": format_results(vector_results)
        })
        
        results["hybrid"].append({
            "query": query,
            "time": hybrid_time,
            "result_count": len(hybrid_results),
            "results": format_results(hybrid_results)
        })
        
        results["hybrid_with_rerank"].append({
            "query": query,
            "time": hybrid_rerank_time,
            "result_count": len(hybrid_rerank_results),
            "results": format_results(hybrid_rerank_results)
        })
        
        # Record timing comparisons
        results["query_times"][query] = {
            "vector_only": vector_time,
            "hybrid": hybrid_time,
            "hybrid_with_rerank": hybrid_rerank_time,
            "hybrid_vs_vector": (hybrid_time / vector_time) if vector_time > 0 else 0,
            "rerank_vs_hybrid": (hybrid_rerank_time / hybrid_time) if hybrid_time > 0 else 0
        }
    
    # Calculate summary statistics
    results["summary"] = calculate_summary_statistics(results)
    
    return results

def format_results(results: List[Dict]) -> List[Dict]:
    """
    Format retrieval results for reporting.
    
    Args:
        results: List of retrieval results
        
    Returns:
        Formatted results
    """
    formatted = []
    
    for i, result in enumerate(results):
        formatted.append({
            "rank": i + 1,
            "chunk_id": result.get("chunk_id", "unknown"),
            "similarity": result.get("similarity", 0),
            "source": result.get("source", "unknown"),
            "content_preview": result.get("content", "")[:200] + "..." 
                if result.get("content") and len(result.get("content", "")) > 200 
                else result.get("content", "No content")
        })
    
    return formatted

def calculate_result_overlap(
    vector_results: List[Dict],
    hybrid_results: List[Dict],
    hybrid_rerank_results: List[Dict]
) -> Dict[str, float]:
    """
    Calculate the overlap between result sets.
    
    Args:
        vector_results: Results from vector-only search
        hybrid_results: Results from hybrid search
        hybrid_rerank_results: Results from hybrid search with reranking
        
    Returns:
        Dictionary of overlap percentages
    """
    # Extract chunk IDs from each result set
    vector_ids = set(r.get("chunk_id", f"unknown_{i}") for i, r in enumerate(vector_results))
    hybrid_ids = set(r.get("chunk_id", f"unknown_{i}") for i, r in enumerate(hybrid_results))
    rerank_ids = set(r.get("chunk_id", f"unknown_{i}") for i, r in enumerate(hybrid_rerank_results))
    
    # Calculate overlaps
    vector_hybrid_overlap = len(vector_ids.intersection(hybrid_ids)) / len(vector_ids) if len(vector_ids) > 0 else 0
    vector_rerank_overlap = len(vector_ids.intersection(rerank_ids)) / len(vector_ids) if len(vector_ids) > 0 else 0
    hybrid_rerank_overlap = len(hybrid_ids.intersection(rerank_ids)) / len(hybrid_ids) if len(hybrid_ids) > 0 else 0
    
    return {
        "vector_hybrid_overlap": vector_hybrid_overlap,
        "vector_rerank_overlap": vector_rerank_overlap,
        "hybrid_rerank_overlap": hybrid_rerank_overlap
    }

def calculate_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate summary statistics from test results.
    
    Args:
        results: Test results
        
    Returns:
        Summary statistics
    """
    # Initialize summary
    summary = {
        "avg_times": {
            "vector_only": 0,
            "hybrid": 0,
            "hybrid_with_rerank": 0
        },
        "avg_result_counts": {
            "vector_only": 0,
            "hybrid": 0,
            "hybrid_with_rerank": 0
        },
        "avg_overlaps": {
            "vector_hybrid_overlap": 0,
            "vector_rerank_overlap": 0,
            "hybrid_rerank_overlap": 0
        }
    }
    
    # Calculate average times
    vector_times = [entry["time"] for entry in results["vector_only"]]
    hybrid_times = [entry["time"] for entry in results["hybrid"]]
    rerank_times = [entry["time"] for entry in results["hybrid_with_rerank"]]
    
    summary["avg_times"]["vector_only"] = sum(vector_times) / len(vector_times) if vector_times else 0
    summary["avg_times"]["hybrid"] = sum(hybrid_times) / len(hybrid_times) if hybrid_times else 0
    summary["avg_times"]["hybrid_with_rerank"] = sum(rerank_times) / len(rerank_times) if rerank_times else 0
    
    # Calculate average result counts
    vector_counts = [entry["result_count"] for entry in results["vector_only"]]
    hybrid_counts = [entry["result_count"] for entry in results["hybrid"]]
    rerank_counts = [entry["result_count"] for entry in results["hybrid_with_rerank"]]
    
    summary["avg_result_counts"]["vector_only"] = sum(vector_counts) / len(vector_counts) if vector_counts else 0
    summary["avg_result_counts"]["hybrid"] = sum(hybrid_counts) / len(hybrid_counts) if hybrid_counts else 0
    summary["avg_result_counts"]["hybrid_with_rerank"] = sum(rerank_counts) / len(rerank_counts) if rerank_counts else 0
    
    # Calculate average overlaps
    overlaps = []
    for i, query_time in enumerate(results["query_times"].values()):
        vector_result = results["vector_only"][i]
        hybrid_result = results["hybrid"][i]
        rerank_result = results["hybrid_with_rerank"][i]
        
        vector_ids = set(r.get("chunk_id", f"unknown_{j}") for j, r in enumerate([r for r in vector_result["results"]]))
        hybrid_ids = set(r.get("chunk_id", f"unknown_{j}") for j, r in enumerate([r for r in hybrid_result["results"]]))
        rerank_ids = set(r.get("chunk_id", f"unknown_{j}") for j, r in enumerate([r for r in rerank_result["results"]]))
        
        # Calculate overlaps
        vector_hybrid_overlap = len(vector_ids.intersection(hybrid_ids)) / len(vector_ids) if len(vector_ids) > 0 else 0
        vector_rerank_overlap = len(vector_ids.intersection(rerank_ids)) / len(vector_ids) if len(vector_ids) > 0 else 0
        hybrid_rerank_overlap = len(hybrid_ids.intersection(rerank_ids)) / len(hybrid_ids) if len(hybrid_ids) > 0 else 0
        
        overlaps.append({
            "vector_hybrid_overlap": vector_hybrid_overlap,
            "vector_rerank_overlap": vector_rerank_overlap,
            "hybrid_rerank_overlap": hybrid_rerank_overlap
        })
    
    # Calculate averages for overlaps
    if overlaps:
        for key in summary["avg_overlaps"].keys():
            values = [overlap[key] for overlap in overlaps]
            summary["avg_overlaps"][key] = sum(values) / len(values) if values else 0
    
    return summary

def main():
    """Run hybrid search performance tests."""
    logger.info("Starting hybrid search performance testing")
    
    # Ensure test document exists
    if not Path(DOCUMENT_PATH).exists():
        logger.error(f"Test document not found: {DOCUMENT_PATH}")
        return
    
    # Process and prepare data
    chunks = process_and_prepare_data(DOCUMENT_PATH)
    
    # Run retrieval tests
    results = test_retrieval_methods(chunks, TEST_QUERIES)
    
    # Save detailed results
    save_json(
        data=results,
        file_path=EVAL_DIR / "hybrid_search_results.json"
    )
    
    # Output summary
    summary = results["summary"]
    
    with open(EVAL_DIR / "hybrid_search_summary.txt", "w") as f:
        f.write("Hybrid Search Performance Summary\n")
        f.write("================================\n\n")
        
        f.write("Average Retrieval Times:\n")
        f.write(f"  Vector-only search: {summary['avg_times']['vector_only']:.4f} seconds\n")
        f.write(f"  Hybrid search: {summary['avg_times']['hybrid']:.4f} seconds\n")
        f.write(f"  Hybrid search with reranking: {summary['avg_times']['hybrid_with_rerank']:.4f} seconds\n\n")
        
        f.write("Average Result Counts:\n")
        f.write(f"  Vector-only search: {summary['avg_result_counts']['vector_only']:.2f} results\n")
        f.write(f"  Hybrid search: {summary['avg_result_counts']['hybrid']:.2f} results\n")
        f.write(f"  Hybrid search with reranking: {summary['avg_result_counts']['hybrid_with_rerank']:.2f} results\n\n")
        
        f.write("Result Set Overlap:\n")
        f.write(f"  Vector vs Hybrid: {summary['avg_overlaps']['vector_hybrid_overlap'] * 100:.2f}%\n")
        f.write(f"  Vector vs Hybrid+Rerank: {summary['avg_overlaps']['vector_rerank_overlap'] * 100:.2f}%\n")
        f.write(f"  Hybrid vs Hybrid+Rerank: {summary['avg_overlaps']['hybrid_rerank_overlap'] * 100:.2f}%\n\n")
        
        f.write("Performance Impact:\n")
        hybrid_vs_vector = summary['avg_times']['hybrid'] / summary['avg_times']['vector_only'] if summary['avg_times']['vector_only'] > 0 else 0
        rerank_vs_hybrid = summary['avg_times']['hybrid_with_rerank'] / summary['avg_times']['hybrid'] if summary['avg_times']['hybrid'] > 0 else 0
        
        f.write(f"  Hybrid search is {hybrid_vs_vector:.2f}x slower than vector-only search\n")
        f.write(f"  Reranking adds {rerank_vs_hybrid:.2f}x overhead to hybrid search\n")
    
    logger.info(f"Results saved to {EVAL_DIR / 'hybrid_search_results.json'}")
    logger.info(f"Summary saved to {EVAL_DIR / 'hybrid_search_summary.txt'}")
    logger.info("Hybrid search performance testing completed")

if __name__ == "__main__":
    main() 