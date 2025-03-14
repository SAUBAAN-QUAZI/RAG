#!/usr/bin/env python
"""
Improved Evaluation of RAG System on LLM Book PDF
------------------------------------------------
This script implements several improvements to the RAG evaluation:
1. Better chunking strategy with larger chunks and more overlap
2. Support for manual relevance judgments
3. Query reformulation to test different phrasings
4. Testing with different embedding models
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

# Add project root to path if running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.document_processing import process_document
from rag.document_processing.document import DocumentChunk
from rag.embedding.service import EmbeddingService
from rag.retrieval.rag_agent import RAGAgent
from rag.retrieval.retriever import Retriever
from rag.evaluation import (
    RAGEvaluator,
    RetrievalTestCase,
    EndToEndTestCase
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Path to the LLM book PDF
LLM_BOOK_PATH = "data/documents/LLM book.pdf"

# Path for evaluation results
EVAL_DIR = Path("data/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Chunking configurations to test
CHUNKING_CONFIGS = [
    {"name": "default", "chunk_size": 1000, "chunk_overlap": 200, "splitter_type": "token"},
    {"name": "large_chunks", "chunk_size": 1500, "chunk_overlap": 300, "splitter_type": "token"},
    {"name": "semantic_chunks", "chunk_size": 1200, "chunk_overlap": 250, "splitter_type": "sentence"}
]

# Embedding models to test
EMBEDDING_MODELS = [
    "text-embedding-3-small",  # Default model
    "text-embedding-3-large"   # More powerful model
]

# Test queries with variations
TEST_QUERIES = {
    "llm_definition": [
        "What are large language models?",
        "Define large language models",
        "Explain what LLMs are"
    ],
    "transformer_architecture": [
        "How does the transformer architecture work?",
        "Explain the transformer architecture",
        "What is the structure of transformer models?"
    ],
    "llm_limitations": [
        "What are the limitations of large language models?",
        "What are the drawbacks of LLMs?",
        "What challenges do large language models face?"
    ],
    "attention_mechanism": [
        "What is the attention mechanism in transformers?",
        "How does attention work in language models?",
        "Explain self-attention in transformers"
    ],
    "llm_training": [
        "How are large language models trained?",
        "What is the training process for LLMs?",
        "Explain how LLMs are pre-trained"
    ]
}

# Manual relevance judgments (to be filled after processing)
# Format: {query_type: {chunk_id: relevance_score}}
# Relevance scores: 0 (not relevant), 1 (somewhat relevant), 2 (highly relevant)
MANUAL_RELEVANCE = {}

def process_llm_book(config: Dict[str, Any]) -> Tuple[Any, List[DocumentChunk]]:
    """
    Process the LLM book PDF with the specified chunking configuration.
    
    Args:
        config: Chunking configuration with chunk_size, chunk_overlap, and splitter_type
        
    Returns:
        Tuple of (document, chunks)
    """
    config_name = config["name"]
    logger.info(f"Processing LLM book PDF with {config_name} chunking configuration")
    logger.info(f"  Chunk size: {config['chunk_size']}, Overlap: {config['chunk_overlap']}, Splitter: {config['splitter_type']}")
    
    # Check if the file exists
    if not Path(LLM_BOOK_PATH).exists():
        logger.error(f"LLM book PDF not found at {LLM_BOOK_PATH}")
        sys.exit(1)
    
    # Process the document
    results = process_document(
        LLM_BOOK_PATH,
        splitter_type=config["splitter_type"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        save_results=True,
        title="LLM Book",
        source_type="pdf",
        category="textbook"
    )
    
    logger.info(f"Document processed. Generated {len(results['chunks'])} chunks.")
    
    # Save chunk IDs and contents for reference
    chunk_ids = [chunk.chunk_id for chunk in results['chunks']]
    chunk_contents = {chunk.chunk_id: chunk.content for chunk in results['chunks']}
    
    output_path = EVAL_DIR / f"llm_book_chunks_{config_name}.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": config,
            "total_chunks": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "chunk_contents": chunk_contents
        }, f, indent=2)
    
    logger.info(f"Chunk information saved to {output_path}")
    
    return results['document'], results['chunks']

def create_test_cases(query_variations: bool = False) -> Dict[str, List[RetrievalTestCase]]:
    """
    Create test cases for the LLM book.
    
    Args:
        query_variations: Whether to include query variations
        
    Returns:
        Dictionary of test cases by query type
    """
    logger.info("Creating test cases for LLM book evaluation")
    
    test_cases_by_type = {}
    
    for query_type, queries in TEST_QUERIES.items():
        test_cases = []
        
        # Use only the first query or all variations
        query_list = queries if query_variations else [queries[0]]
        
        for query in query_list:
            # Create test case with empty relevant_chunk_ids (to be filled later)
            test_case = RetrievalTestCase(
                query=query,
                relevant_chunk_ids=[],
                description=f"{query_type} query"
            )
            test_cases.append(test_case)
        
        test_cases_by_type[query_type] = test_cases
    
    return test_cases_by_type

def find_candidate_chunks(
    rag_agent: RAGAgent,
    test_cases_by_type: Dict[str, List[RetrievalTestCase]],
    top_k: int = 10
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Find candidate relevant chunks for each test case query.
    
    Args:
        rag_agent: The RAG agent to use for retrieval
        test_cases_by_type: Dictionary of test cases by query type
        top_k: Number of top chunks to retrieve
        
    Returns:
        Dictionary of candidate chunks by query type and query
    """
    logger.info(f"Finding candidate chunks for test queries (top_k={top_k})")
    
    candidate_chunks = {}
    
    for query_type, test_cases in test_cases_by_type.items():
        candidate_chunks[query_type] = {}
        
        for test_case in test_cases:
            query = test_case.query
            logger.info(f"Processing query: {query}")
            
            # Retrieve relevant chunks
            start_time = time.time()
            results = rag_agent.retriever.retrieve(query, top_k=top_k)
            retrieval_time = time.time() - start_time
            
            # Store results with retrieval time
            candidate_chunks[query_type][query] = {
                "results": results,
                "retrieval_time": retrieval_time
            }
            
            # Log the found chunks
            logger.info(f"Found {len(results)} candidate chunks for query: {query} in {retrieval_time:.2f}s")
            for i, result in enumerate(results):
                # Check for different possible score keys
                score = None
                for key in ['score', 'similarity', 'distance', 'relevance']:
                    if key in result:
                        score = result[key]
                        break
                
                score_str = f", Score: {score:.4f}" if score is not None else ""
                logger.info(f"  {i+1}. Chunk ID: {result['chunk_id']}{score_str}")
                logger.info(f"     Preview: {result['content'][:150]}...")
    
    # Save candidate chunks for manual review
    output_path = EVAL_DIR / "llm_book_candidate_chunks.json"
    
    # Convert to serializable format
    serializable_candidates = {}
    for query_type, queries in candidate_chunks.items():
        serializable_candidates[query_type] = {}
        for query, data in queries.items():
            serializable_candidates[query_type][query] = {
                "results": [
                    {
                        "chunk_id": r["chunk_id"],
                        "content": r["content"][:500] + ("..." if len(r["content"]) > 500 else ""),
                        "score": next((r[k] for k in ['score', 'similarity', 'distance', 'relevance'] if k in r), None)
                    }
                    for r in data["results"]
                ],
                "retrieval_time": data["retrieval_time"]
            }
    
    with open(output_path, "w") as f:
        json.dump(serializable_candidates, f, indent=2)
    
    logger.info(f"Candidate chunks saved to {output_path} for manual review")
    
    return candidate_chunks

def load_or_create_manual_relevance(candidate_chunks: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load existing manual relevance judgments or create a template for them.
    
    Args:
        candidate_chunks: Dictionary of candidate chunks by query type and query
        
    Returns:
        Dictionary of relevance judgments by query type, query, and chunk_id
    """
    manual_relevance_path = EVAL_DIR / "llm_book_manual_relevance.json"
    
    if manual_relevance_path.exists():
        logger.info(f"Loading existing manual relevance judgments from {manual_relevance_path}")
        with open(manual_relevance_path, "r") as f:
            return json.load(f)
    
    # Create template for manual relevance judgments
    logger.info("Creating template for manual relevance judgments")
    manual_relevance = {}
    
    for query_type, queries in candidate_chunks.items():
        manual_relevance[query_type] = {}
        
        for query, data in queries.items():
            manual_relevance[query_type][query] = {}
            
            for result in data["results"]:
                # Default relevance score is 0 (not relevant)
                manual_relevance[query_type][query][result["chunk_id"]] = 0
    
    # Save template
    with open(manual_relevance_path, "w") as f:
        json.dump(manual_relevance, f, indent=2)
    
    logger.info(f"Manual relevance template saved to {manual_relevance_path}")
    logger.info("Please edit this file to assign relevance scores (0=not relevant, 1=somewhat relevant, 2=highly relevant)")
    
    return manual_relevance

def create_test_cases_with_relevance(
    manual_relevance: Dict[str, Dict[str, Dict[str, float]]],
    relevance_threshold: float = 1.0
) -> Dict[str, List[RetrievalTestCase]]:
    """
    Create test cases with manually judged relevant chunks.
    
    Args:
        manual_relevance: Dictionary of relevance judgments
        relevance_threshold: Minimum relevance score to consider a chunk relevant
        
    Returns:
        Dictionary of test cases by query type
    """
    logger.info(f"Creating test cases with manual relevance judgments (threshold={relevance_threshold})")
    
    test_cases_by_type = {}
    
    for query_type, queries in manual_relevance.items():
        test_cases = []
        
        for query, chunk_scores in queries.items():
            # Filter chunks by relevance threshold
            relevant_chunk_ids = [
                chunk_id for chunk_id, score in chunk_scores.items()
                if score >= relevance_threshold
            ]
            
            # Create test case with manually judged relevant chunks
            test_case = RetrievalTestCase(
                query=query,
                relevant_chunk_ids=relevant_chunk_ids,
                description=f"{query_type} query"
            )
            
            test_cases.append(test_case)
            logger.info(f"Query: {query}")
            logger.info(f"  Relevant chunks: {len(relevant_chunk_ids)}")
        
        test_cases_by_type[query_type] = test_cases
    
    return test_cases_by_type

def evaluate_system(
    rag_agent: RAGAgent,
    test_cases_by_type: Dict[str, List[RetrievalTestCase]],
    config_name: str,
    embedding_model: str
) -> Dict[str, Any]:
    """
    Evaluate the RAG system using the test cases.
    
    Args:
        rag_agent: The RAG agent to use for evaluation
        test_cases_by_type: Dictionary of test cases by query type
        config_name: Name of the chunking configuration
        embedding_model: Name of the embedding model
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"Evaluating RAG system performance with {config_name} chunks and {embedding_model}")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_agent)
    
    # Flatten test cases for overall evaluation
    all_test_cases = []
    for test_cases in test_cases_by_type.values():
        all_test_cases.extend(test_cases)
    
    # Evaluate overall retrieval performance
    logger.info(f"Running overall retrieval evaluation with {len(all_test_cases)} test cases")
    overall_results = evaluator.evaluate_retrieval(all_test_cases)
    
    # Evaluate per query type
    per_type_results = {}
    for query_type, test_cases in test_cases_by_type.items():
        if not test_cases:
            continue
            
        logger.info(f"Running retrieval evaluation for {query_type} with {len(test_cases)} test cases")
        type_results = evaluator.evaluate_retrieval(test_cases)
        per_type_results[query_type] = type_results
    
    # Print overall results
    logger.info("Overall Retrieval Performance:")
    for metric, value in overall_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results = {
        "config_name": config_name,
        "embedding_model": embedding_model,
        "overall": overall_results._asdict(),
        "per_type": {query_type: results._asdict() for query_type, results in per_type_results.items()}
    }
    
    results_path = EVAL_DIR / f"llm_book_results_{config_name}_{embedding_model.replace('-', '_')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")
    
    return results

def compare_configurations(results_files: List[Path]) -> None:
    """
    Compare results across different configurations.
    
    Args:
        results_files: List of result file paths to compare
    """
    logger.info(f"Comparing {len(results_files)} configurations")
    
    comparison = {
        "configurations": [],
        "overall_metrics": {},
        "per_type_metrics": {}
    }
    
    for file_path in results_files:
        with open(file_path, "r") as f:
            results = json.load(f)
        
        config_name = results["config_name"]
        embedding_model = results["embedding_model"]
        config_id = f"{config_name}_{embedding_model}"
        
        comparison["configurations"].append({
            "id": config_id,
            "config_name": config_name,
            "embedding_model": embedding_model
        })
        
        # Extract overall metrics
        for metric, value in results["overall"]["metrics"].items():
            if metric not in comparison["overall_metrics"]:
                comparison["overall_metrics"][metric] = {}
            comparison["overall_metrics"][metric][config_id] = value
        
        # Extract per-type metrics
        for query_type, type_results in results["per_type"].items():
            if query_type not in comparison["per_type_metrics"]:
                comparison["per_type_metrics"][query_type] = {}
            
            for metric, value in type_results["metrics"].items():
                metric_key = f"{query_type}_{metric}"
                if metric_key not in comparison["per_type_metrics"][query_type]:
                    comparison["per_type_metrics"][query_type][metric] = {}
                comparison["per_type_metrics"][query_type][metric][config_id] = value
    
    # Print comparison
    logger.info("Configuration Comparison:")
    logger.info("------------------------")
    
    logger.info("Overall Metrics:")
    for metric, values in comparison["overall_metrics"].items():
        logger.info(f"  {metric}:")
        for config_id, value in values.items():
            logger.info(f"    {config_id}: {value:.4f}")
    
    logger.info("Per-Type Metrics:")
    for query_type, metrics in comparison["per_type_metrics"].items():
        logger.info(f"  {query_type}:")
        for metric, values in metrics.items():
            logger.info(f"    {metric}:")
            for config_id, value in values.items():
                logger.info(f"      {config_id}: {value:.4f}")
    
    # Save comparison
    comparison_path = EVAL_DIR / "llm_book_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison saved to {comparison_path}")

def main():
    """Run the improved LLM book evaluation."""
    logger.info("Starting improved LLM book evaluation")
    
    # Check if evaluation directory exists
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process the LLM book with different chunking configurations
    processed_configs = []
    chunks_by_config = {}
    
    for config in CHUNKING_CONFIGS:
        document, chunks = process_llm_book(config)
        chunks_by_config[config["name"]] = chunks
        processed_configs.append(config["name"])
    
    # Test with different embedding models and chunking configurations
    results_files = []
    
    for config_name in processed_configs:
        chunks = chunks_by_config[config_name]
        
        for embedding_model in EMBEDDING_MODELS:
            # Initialize embedding service with the specified model
            embedding_service = EmbeddingService(model=embedding_model)
            
            # Initialize retriever with the embedding service
            retriever = Retriever(embedding_service=embedding_service)
            
            # Add document chunks to the retriever
            logger.info(f"Adding {len(chunks)} chunks to retriever")
            retriever.add_chunks(chunks)
            
            # Initialize RAG agent with the retriever
            rag_agent = RAGAgent(retriever=retriever)
            
            # Find candidate chunks for manual review
            test_cases_by_type = create_test_cases(query_variations=True)
            candidate_chunks = find_candidate_chunks(rag_agent, test_cases_by_type)
            
            # Load or create manual relevance judgments
            manual_relevance = load_or_create_manual_relevance(candidate_chunks)
            
            # Create test cases with manual relevance judgments
            test_cases_with_relevance = create_test_cases_with_relevance(manual_relevance)
            
            # Evaluate system performance
            results = evaluate_system(
                rag_agent,
                test_cases_with_relevance,
                config_name,
                embedding_model
            )
            
            # Add results file to the list for comparison
            results_path = EVAL_DIR / f"llm_book_results_{config_name}_{embedding_model.replace('-', '_')}.json"
            results_files.append(results_path)
    
    # Compare configurations
    if len(results_files) > 1:
        compare_configurations(results_files)
    
    logger.info("Improved LLM book evaluation completed")
    logger.info("\nNext steps:")
    logger.info("1. Review the candidate chunks in data/evaluation/llm_book_candidate_chunks.json")
    logger.info("2. Edit data/evaluation/llm_book_manual_relevance.json to assign relevance scores")
    logger.info("3. Re-run this script to evaluate with manual relevance judgments")

if __name__ == "__main__":
    main() 