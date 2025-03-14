#!/usr/bin/env python
"""
Testing Enhanced Retrieval Performance
--------------------------------------
This script evaluates the performance of the enhanced retrieval system with:
1. SemanticTextSplitter for smarter document chunking
2. Hybrid search combining vector and keyword search
3. Cross-encoder reranking for improved result ranking
4. Optimized vector store settings

The script runs tests with different configurations and compares results.
"""

import os
import sys
import logging
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path if running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.document_processing import process_document
from rag.document_processing.document import DocumentChunk
from rag.document_processing.splitters import get_text_splitter
from rag.embedding.service import EmbeddingService
from rag.retrieval.rag_agent import RAGAgent
from rag.retrieval.retriever import Retriever
from rag.evaluation import (
    RAGEvaluator,
    RetrievalTestCase,
)
from rag.utils import save_json, load_json

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

# Test queries for evaluation
TEST_QUERIES = [
    "What are the key components of a transformer architecture?",
    "How do large language models handle context?",
    "What are the limitations of current large language models?",
    "Explain the attention mechanism in transformers",
    "How does fine-tuning improve LLM performance?",
    "What is the role of tokenization in large language models?",
    "How do prompt engineering techniques work?",
    "What are embeddings and how are they used in LLMs?",
    "Explain the concept of zero-shot learning in language models",
    "What ethical concerns are associated with large language models?"
]

# Configurations to test
CONFIGS = [
    {
        "name": "baseline",
        "splitter_type": "token",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "enable_hybrid_search": False,
        "enable_reranking": False,
        "description": "Baseline configuration with token splitter and basic vector search"
    },
    {
        "name": "semantic_splitter",
        "splitter_type": "semantic",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "enable_hybrid_search": False,
        "enable_reranking": False,
        "description": "Semantic text splitter with basic vector search"
    },
    {
        "name": "hybrid_search",
        "splitter_type": "token",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "enable_hybrid_search": True,
        "enable_reranking": False,
        "description": "Token splitter with hybrid search (vector + keyword)"
    },
    {
        "name": "reranking",
        "splitter_type": "token",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "enable_hybrid_search": False,
        "enable_reranking": True,
        "description": "Token splitter with cross-encoder reranking"
    },
    {
        "name": "full_enhanced",
        "splitter_type": "semantic",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "enable_hybrid_search": True,
        "enable_reranking": True,
        "description": "Full enhanced retrieval with semantic splitter, hybrid search, and reranking"
    }
]

def process_test_document(doc_path: str, config: Dict[str, Any]) -> Tuple[str, List[DocumentChunk]]:
    """
    Process the test document with the specified chunking configuration.
    
    Args:
        doc_path: Path to the document
        config: Chunking configuration
        
    Returns:
        Tuple containing the document ID and list of chunks
    """
    logger.info(f"Processing document with config: {config['name']}")
    
    # Process document
    result = process_document(
        file_path=doc_path,
        splitter_type=config["splitter_type"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    
    # Extract document and chunks from result
    document = result["document"]
    chunks = result["chunks"]
    
    logger.info(f"Document {document.doc_id} processed into {len(chunks)} chunks")
    
    # Save chunks for inspection
    chunks_data = [
        {
            "chunk_id": chunk.metadata.get("chunk_id", str(uuid.uuid4())),
            "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    
    save_json(
        data=chunks_data,
        file_path=EVAL_DIR / f"chunks_{config['name']}.json"
    )
    
    return document.doc_id, chunks

def initialize_retriever(config: Dict[str, Any], collection_name: str) -> Retriever:
    """
    Initialize a retriever with the specified configuration.
    
    Args:
        config: Retriever configuration
        collection_name: Name for the vector store collection
        
    Returns:
        Initialized Retriever instance
    """
    logger.info(f"Initializing retriever with config: {config['name']}")
    
    # Create embedding service
    embedding_service = EmbeddingService()
    
    # Create retriever with configuration settings
    retriever = Retriever(
        embedding_service=embedding_service,
        collection_name=collection_name,
        enable_hybrid_search=config["enable_hybrid_search"],
        enable_reranking=config["enable_reranking"],
    )
    
    return retriever

def run_retrieval_test(
    config: Dict[str, Any],
    queries: List[str],
    doc_path: str
) -> Dict[str, Any]:
    """
    Run retrieval test with a specific configuration.
    
    Args:
        config: Configuration to test
        queries: List of test queries
        doc_path: Path to document
        
    Returns:
        Dictionary of test results
    """
    start_time = time.time()
    
    # Create unique collection name for this test
    collection_name = f"test_{config['name']}_{int(time.time())}"
    
    # Initialize retriever
    retriever = initialize_retriever(config, collection_name)
    
    # Process document and add chunks to retriever
    doc_id, chunks = process_test_document(doc_path, config)
    retriever.add_chunks(chunks)
    
    # Create test cases
    test_cases = []
    
    for query in queries:
        test_cases.append(RetrievalTestCase(
            query=query,
            relevant_chunk_ids=[]  # We'll evaluate based on returned content relevance
        ))
    
    # Run tests for each query
    results = []
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"Testing query {i+1}/{len(test_cases)}: {test_case.query}")
        
        # Time the retrieval operation
        query_start_time = time.time()
        retrieved_results = retriever.retrieve(query=test_case.query, top_k=5)
        query_time = time.time() - query_start_time
        
        # Record results
        query_result = {
            "query": test_case.query,
            "time_seconds": query_time,
            "num_results": len(retrieved_results),
            "results": []
        }
        
        # Record detailed results
        for j, result in enumerate(retrieved_results):
            query_result["results"].append({
                "rank": j + 1,
                "chunk_id": result.get("chunk_id", "unknown"),
                "similarity": result.get("similarity", 0),
                "source": result.get("source", "unknown"),
                "content_preview": result.get("content", "")[:200] + "..." 
                    if result.get("content") and len(result.get("content", "")) > 200 
                    else result.get("content", "No content")
            })
        
        results.append(query_result)
    
    # Calculate average retrieval time
    avg_retrieval_time = sum(r["time_seconds"] for r in results) / len(results)
    
    # Save detailed results
    test_results = {
        "config": config,
        "document_id": doc_id,
        "num_chunks": len(chunks),
        "avg_retrieval_time": avg_retrieval_time,
        "query_results": results,
        "timestamp": time.time(),
    }
    
    save_json(
        data=test_results,
        file_path=EVAL_DIR / f"retrieval_test_{config['name']}.json"
    )
    
    # Calculate success metrics
    avg_results = sum(len(r["results"]) for r in results) / len(results)
    
    total_time = time.time() - start_time
    logger.info(f"Test completed in {total_time:.2f} seconds for configuration: {config['name']}")
    logger.info(f"Average retrieval time: {avg_retrieval_time:.4f} seconds")
    logger.info(f"Average results per query: {avg_results:.2f}")
    
    return test_results

def compare_results(results: List[Dict[str, Any]]) -> None:
    """
    Compare results from different configurations.
    
    Args:
        results: List of test results for different configurations
    """
    logger.info("Comparing retrieval results across configurations")
    
    # Create comparison data
    comparison_data = []
    
    for result in results:
        config = result["config"]
        comparison_data.append({
            "Configuration": config["name"],
            "Description": config["description"],
            "Avg Retrieval Time (s)": result["avg_retrieval_time"],
            "Chunks Created": result["num_chunks"],
        })
    
    # Create DataFrame for analysis
    df = pd.DataFrame(comparison_data)
    
    # Save comparison to CSV
    df.to_csv(EVAL_DIR / "retrieval_comparison.csv", index=False)
    logger.info(f"Comparison saved to {EVAL_DIR / 'retrieval_comparison.csv'}")
    
    # Create comparison chart
    plt.figure(figsize=(10, 6))
    plt.bar(df["Configuration"], df["Avg Retrieval Time (s)"])
    plt.title("Average Retrieval Time by Configuration")
    plt.xlabel("Configuration")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "retrieval_time_comparison.png")
    logger.info(f"Comparison chart saved to {EVAL_DIR / 'retrieval_time_comparison.png'}")

def main():
    """Run the retrieval performance tests."""
    logger.info("Starting enhanced retrieval performance testing")
    
    # Ensure test document exists
    if not Path(DOCUMENT_PATH).exists():
        logger.error(f"Test document not found: {DOCUMENT_PATH}")
        logger.info("Please provide a valid document path or use a different document.")
        return
    
    # Run tests for each configuration
    results = []
    
    for config in CONFIGS:
        logger.info(f"Testing configuration: {config['name']}")
        result = run_retrieval_test(config, TEST_QUERIES, DOCUMENT_PATH)
        results.append(result)
    
    # Compare results across configurations
    compare_results(results)
    
    logger.info("Enhanced retrieval performance testing completed")

if __name__ == "__main__":
    main() 