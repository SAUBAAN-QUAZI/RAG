#!/usr/bin/env python
"""
Example: Evaluating RAG System Performance
-----------------------------------------
This script demonstrates how to use the RAG evaluation framework
to measure the performance of the system.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path if running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retrieval.rag_agent import RAGAgent
from rag.evaluation import (
    RAGEvaluator,
    RetrievalTestCase,
    GenerationTestCase,
    EndToEndTestCase
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Run RAG evaluation example."""
    
    logger.info("Initializing RAG agent")
    rag_agent = RAGAgent()
    
    logger.info("Initializing evaluator")
    evaluator = RAGEvaluator(rag_agent)
    
    # Example: Create and save a test dataset
    # In a real scenario, you would define relevant chunk IDs based on your indexed documents
    retrieval_test_cases = [
        RetrievalTestCase(
            query="What are large language models?",
            relevant_chunk_ids=["abc123", "def456"],  # Replace with actual chunk IDs
            description="Basic LLM definition query"
        ),
        RetrievalTestCase(
            query="How does retrieval-augmented generation work?",
            relevant_chunk_ids=["ghi789", "jkl012"],  # Replace with actual chunk IDs
            description="RAG explanation query"
        ),
    ]
    
    generation_test_cases = [
        GenerationTestCase(
            query="Explain the concept of embeddings",
            context="Embeddings are dense vector representations of text that capture semantic meaning. "
                   "They map words or phrases to points in a high-dimensional space where similar "
                   "items are closer together. This allows machines to understand relationships "
                   "between different pieces of text.",
            expected_answer="Embeddings are vector representations that capture the meaning of text, "
                           "placing semantically similar concepts close together in a high-dimensional space.",
            description="Embeddings explanation"
        ),
    ]
    
    end_to_end_test_cases = [
        EndToEndTestCase(
            query="What is the difference between semantic and keyword search?",
            relevant_chunk_ids=["mno345", "pqr678"],  # Replace with actual chunk IDs
            expected_answer="Semantic search understands the meaning and context of a query, "
                           "while keyword search only matches specific terms.",
            description="Search comparison query"
        ),
    ]
    
    # Create and save the test dataset
    dataset_path = Path("data/evaluation/test_dataset.json")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    evaluator.create_test_dataset(
        retrieval_cases=retrieval_test_cases,
        generation_cases=generation_test_cases,
        end_to_end_cases=end_to_end_test_cases,
        file_path=dataset_path
    )
    
    logger.info(f"Test dataset saved to {dataset_path}")
    
    # In a real implementation, you would have documents already indexed
    # and test against actual chunk IDs from those documents
    
    # Example: Evaluate retrieval (with placeholder data)
    # Note: This will likely return low scores without real indexed data
    logger.info("Running retrieval evaluation")
    
    try:
        retrieval_results = evaluator.evaluate_retrieval(retrieval_test_cases)
        
        logger.info("Retrieval Performance:")
        for metric, value in retrieval_results.metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        # Save results
        results_path = Path("data/evaluation/retrieval_results.json")
        evaluator.save_results(retrieval_results, results_path)
        logger.info(f"Retrieval results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error in retrieval evaluation: {e}")
        logger.info("This example may fail without actual indexed documents")
    
    logger.info("Evaluation example completed")
    
if __name__ == "__main__":
    main() 