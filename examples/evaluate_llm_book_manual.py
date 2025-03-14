#!/usr/bin/env python
"""
Evaluation of RAG System with Manual Relevance Judgments
-------------------------------------------------------
This script evaluates the RAG system using manually assigned relevance judgments.
Run this after editing the manual_relevance.json file.
"""

import sys
import logging
import json
from pathlib import Path
import time

# Add project root to path if running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retrieval.rag_agent import RAGAgent
from rag.evaluation import (
    RAGEvaluator,
    RetrievalTestCase
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Path for evaluation results
EVAL_DIR = Path("data/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Path to manual relevance judgments
MANUAL_RELEVANCE_PATH = EVAL_DIR / "llm_book_manual_relevance.json"

# Path to chunk information
CHUNKS_PATH = EVAL_DIR / "llm_book_chunks_improved.json"

def load_manual_relevance():
    """Load manual relevance judgments."""
    if not MANUAL_RELEVANCE_PATH.exists():
        logger.error(f"Manual relevance judgments not found at {MANUAL_RELEVANCE_PATH}")
        logger.error("Please run evaluate_llm_book_initial.py first to generate the template")
        sys.exit(1)
    
    logger.info(f"Loading manual relevance judgments from {MANUAL_RELEVANCE_PATH}")
    with open(MANUAL_RELEVANCE_PATH, "r") as f:
        manual_relevance = json.load(f)
    
    return manual_relevance

def load_chunks():
    """Load chunk information."""
    if not CHUNKS_PATH.exists():
        logger.error(f"Chunk information not found at {CHUNKS_PATH}")
        logger.error("Please run evaluate_llm_book_initial.py first to process the document")
        sys.exit(1)
    
    logger.info(f"Loading chunk information from {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "r") as f:
        chunk_info = json.load(f)
    
    return chunk_info

def create_test_cases_with_relevance(manual_relevance, relevance_threshold=1.0):
    """
    Create test cases with manually judged relevant chunks.
    
    Args:
        manual_relevance: Dictionary of relevance judgments
        relevance_threshold: Minimum relevance score to consider a chunk relevant
        
    Returns:
        List of test cases
    """
    logger.info(f"Creating test cases with manual relevance judgments (threshold={relevance_threshold})")
    
    test_cases = []
    
    for query, chunk_scores in manual_relevance.items():
        # Filter chunks by relevance threshold
        relevant_chunk_ids = [
            chunk_id for chunk_id, score in chunk_scores.items()
            if score >= relevance_threshold
        ]
        
        # Skip queries with no relevant chunks
        if not relevant_chunk_ids:
            logger.warning(f"Query has no relevant chunks with threshold {relevance_threshold}: {query}")
            continue
        
        # Create test case with manually judged relevant chunks
        test_case = RetrievalTestCase(
            query=query,
            relevant_chunk_ids=relevant_chunk_ids,
            description=f"Manual relevance query"
        )
        
        test_cases.append(test_case)
        logger.info(f"Query: {query}")
        logger.info(f"  Relevant chunks: {len(relevant_chunk_ids)}")
    
    return test_cases

def evaluate_system(rag_agent, test_cases):
    """
    Evaluate the RAG system using the test cases.
    
    Args:
        rag_agent: The RAG agent to use for evaluation
        test_cases: List of test cases
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating RAG system performance with manual relevance judgments")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_agent)
    
    # Evaluate retrieval performance
    logger.info(f"Running retrieval evaluation with {len(test_cases)} test cases")
    retrieval_results = evaluator.evaluate_retrieval(test_cases)
    
    # Print results
    logger.info("Retrieval Performance (Manual Relevance):")
    for metric, value in retrieval_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results_path = EVAL_DIR / "llm_book_manual_relevance_results.json"
    evaluator.save_results(retrieval_results, results_path)
    logger.info(f"Manual relevance results saved to {results_path}")
    
    return retrieval_results

def compare_with_auto_relevance():
    """Compare manual relevance results with auto-relevance results."""
    auto_path = EVAL_DIR / "llm_book_auto_relevance_results.json"
    manual_path = EVAL_DIR / "llm_book_manual_relevance_results.json"
    
    if not auto_path.exists() or not manual_path.exists():
        logger.warning("Cannot compare results: one or both result files missing")
        return
    
    logger.info("Comparing auto-relevance and manual relevance results")
    
    with open(auto_path, "r") as f:
        auto_results = json.load(f)
    
    with open(manual_path, "r") as f:
        manual_results = json.load(f)
    
    # Compare metrics
    logger.info("Metric Comparison:")
    for metric in auto_results.get("metrics", {}):
        auto_value = auto_results["metrics"].get(metric, 0)
        manual_value = manual_results["metrics"].get(metric, 0)
        difference = manual_value - auto_value
        
        logger.info(f"  {metric}:")
        logger.info(f"    Auto: {auto_value:.4f}")
        logger.info(f"    Manual: {manual_value:.4f}")
        logger.info(f"    Difference: {difference:.4f} ({difference*100:.1f}%)")
    
    # Save comparison
    comparison = {
        "auto": auto_results,
        "manual": manual_results,
        "differences": {
            metric: {
                "auto": auto_results["metrics"].get(metric, 0),
                "manual": manual_results["metrics"].get(metric, 0),
                "difference": manual_results["metrics"].get(metric, 0) - auto_results["metrics"].get(metric, 0),
                "percent_difference": (manual_results["metrics"].get(metric, 0) - auto_results["metrics"].get(metric, 0)) * 100
            }
            for metric in auto_results.get("metrics", {})
        }
    }
    
    comparison_path = EVAL_DIR / "llm_book_relevance_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison saved to {comparison_path}")

def main():
    """Run the evaluation with manual relevance judgments."""
    logger.info("Starting evaluation with manual relevance judgments")
    
    # Check if evaluation directory exists
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load manual relevance judgments
    manual_relevance = load_manual_relevance()
    
    # Load chunk information
    chunk_info = load_chunks()
    
    # Create test cases with manual relevance judgments
    test_cases = create_test_cases_with_relevance(manual_relevance)
    
    if not test_cases:
        logger.error("No test cases created. Please assign relevance scores in the manual relevance file.")
        sys.exit(1)
    
    # Initialize RAG agent
    logger.info("Initializing RAG agent")
    rag_agent = RAGAgent()
    
    # Evaluate system performance
    results = evaluate_system(rag_agent, test_cases)
    
    # Compare with auto-relevance results
    compare_with_auto_relevance()
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Test cases: {len(test_cases)}")
    logger.info("Manual Relevance Performance:")
    for metric, value in results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nNext steps:")
    logger.info("1. Review the comparison between auto and manual relevance")
    logger.info("2. Consider adjusting chunking parameters or embedding model for better performance")
    logger.info("3. Run the full evaluation script with different configurations")
    
    logger.info("Manual relevance evaluation completed")

if __name__ == "__main__":
    main() 