#!/usr/bin/env python
"""
Simplified Evaluation of RAG System on LLM Book PDF
--------------------------------------------------
This script evaluates the RAG system's performance on a few test cases
using the LLM book PDF that has already been processed.
"""

import sys
import logging
from pathlib import Path
import json

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

def main():
    """Run a simplified evaluation on the LLM book."""
    logger.info("Starting simplified LLM book evaluation")
    
    # Initialize RAG agent
    logger.info("Initializing RAG agent")
    rag_agent = RAGAgent()
    
    # Create a few test cases
    logger.info("Creating test cases")
    test_cases = [
        RetrievalTestCase(
            query="What are large language models?",
            relevant_chunk_ids=[],  # Will be filled automatically
            description="Basic LLM definition query"
        ),
        RetrievalTestCase(
            query="How does the transformer architecture work?",
            relevant_chunk_ids=[],  # Will be filled automatically
            description="Transformer architecture explanation"
        ),
        RetrievalTestCase(
            query="What are the limitations of large language models?",
            relevant_chunk_ids=[],  # Will be filled automatically
            description="LLM limitations query"
        )
    ]
    
    # Find relevant chunks for each test case
    logger.info("Finding relevant chunks for test queries")
    updated_test_cases = []
    
    for test_case in test_cases:
        logger.info(f"Processing query: {test_case.query}")
        
        # Retrieve relevant chunks
        results = rag_agent.retriever.retrieve(test_case.query, top_k=5)
        
        # Extract chunk IDs
        chunk_ids = [result["chunk_id"] for result in results]
        
        # Create updated test case with found chunk IDs
        updated_test_case = RetrievalTestCase(
            query=test_case.query,
            relevant_chunk_ids=chunk_ids,
            description=test_case.description
        )
        
        updated_test_cases.append(updated_test_case)
        
        # Log the found chunks
        logger.info(f"Found {len(chunk_ids)} relevant chunks for query: {test_case.query}")
        for i, result in enumerate(results):
            # Check for different possible score keys
            score = None
            for key in ['score', 'similarity', 'distance', 'relevance']:
                if key in result:
                    score = result[key]
                    break
            
            score_str = f", Score: {score:.4f}" if score is not None else ""
            logger.info(f"  {i+1}. Chunk ID: {result['chunk_id']}{score_str}")
            logger.info(f"     Preview: {result['content'][:100]}...")
    
    # Evaluate system performance
    logger.info("Evaluating RAG system performance")
    evaluator = RAGEvaluator(rag_agent)
    retrieval_results = evaluator.evaluate_retrieval(updated_test_cases)
    
    # Print results
    logger.info("Retrieval Performance:")
    for metric, value in retrieval_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results_path = EVAL_DIR / "llm_book_simple_results.json"
    evaluator.save_results(retrieval_results, results_path)
    logger.info(f"Results saved to {results_path}")
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Test cases: {len(updated_test_cases)}")
    logger.info("Retrieval Performance:")
    for metric, value in retrieval_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nNote: This evaluation uses automatically identified 'relevant' chunks.")
    logger.info("For a more accurate evaluation, manually identify truly relevant chunks.")
    
    logger.info("Simplified evaluation completed")

if __name__ == "__main__":
    main() 