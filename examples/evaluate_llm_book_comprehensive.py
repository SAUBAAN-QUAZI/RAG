#!/usr/bin/env python
"""
Comprehensive Evaluation of RAG System
-------------------------------------
This script evaluates the RAG system with different chunking strategies
and embedding models to identify the optimal configuration.
"""

import sys
import logging
import json
import time
import itertools
from pathlib import Path
import os

# Add project root to path if running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retrieval.rag_agent import RAGAgent
from rag.document_processing.processor import DocumentProcessor
from rag.document_processing.splitters import TextSplitter
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

# Path to LLM book PDF
PDF_PATH = Path("data/documents/LLM book.pdf")

# Test queries
TEST_QUERIES = [
    "What are large language models?",
    "How does the transformer architecture work?",
    "What are the limitations of large language models?",
    "How can large language models be fine-tuned?",
    "What is the difference between pre-training and fine-tuning?",
    "How do large language models handle context?",
    "What ethical concerns exist with large language models?",
    "How do large language models generate text?",
    "What is the role of attention mechanisms in transformers?",
    "How can hallucinations in large language models be reduced?"
]

# Chunking configurations to test
CHUNKING_CONFIGS = [
    {"name": "small_chunks", "chunk_size": 500, "chunk_overlap": 50, "splitter_type": "token"},
    {"name": "medium_chunks", "chunk_size": 1000, "chunk_overlap": 200, "splitter_type": "token"},
    {"name": "large_chunks", "chunk_size": 1500, "chunk_overlap": 300, "splitter_type": "token"},
    {"name": "sentence_chunks", "chunk_size": 40, "chunk_overlap": 5, "splitter_type": "sentence"}
]

# Embedding models to test
EMBEDDING_MODELS = [
    {"name": "text-embedding-ada-002", "dimensions": 1536},
    {"name": "text-embedding-3-small", "dimensions": 1536},
    {"name": "text-embedding-3-large", "dimensions": 3072}
]

def process_document(config):
    """
    Process the LLM book PDF with the given chunking configuration.
    
    Args:
        config: Dictionary with chunking configuration
        
    Returns:
        Tuple of (document_id, chunks)
    """
    config_name = config["name"]
    logger.info(f"Processing document with config: {config_name}")
    
    # Check if PDF exists
    if not PDF_PATH.exists():
        logger.error(f"PDF not found at {PDF_PATH}")
        sys.exit(1)
    
    # Initialize document processor with the specified configuration
    processor = DocumentProcessor()
    
    # Configure text splitter using the get_text_splitter function
    from rag.document_processing.splitters import get_text_splitter
    splitter = get_text_splitter(
        splitter_type=config["splitter_type"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    
    # Process document
    start_time = time.time()
    document_id, chunks = processor.process_file(
        str(PDF_PATH),
        text_splitter=splitter
    )
    processing_time = time.time() - start_time
    
    # Save chunk information
    chunk_info = {
        "document_id": document_id,
        "config": config,
        "num_chunks": len(chunks),
        "processing_time": processing_time,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
    }
    
    output_path = EVAL_DIR / f"llm_book_chunks_{config_name}.json"
    with open(output_path, "w") as f:
        json.dump(chunk_info, f, indent=2)
    
    logger.info(f"Processed document with {len(chunks)} chunks using {config_name} config")
    logger.info(f"Chunk information saved to {output_path}")
    
    return document_id, chunks

def create_test_cases():
    """
    Create test cases from the test queries.
    
    Returns:
        List of test cases
    """
    logger.info("Creating test cases")
    
    test_cases = []
    
    for query in TEST_QUERIES:
        test_case = RetrievalTestCase(
            query=query,
            relevant_chunk_ids=[],  # Will be populated during evaluation
            description=f"Auto-relevance query"
        )
        test_cases.append(test_case)
    
    return test_cases

def evaluate_configuration(config, embedding_model, rag_agent, test_cases):
    """
    Evaluate a specific configuration of chunking and embedding model.
    
    Args:
        config: Dictionary with chunking configuration
        embedding_model: Dictionary with embedding model information
        rag_agent: The RAG agent to use for evaluation
        test_cases: List of test cases
        
    Returns:
        Evaluation results
    """
    config_name = config["name"]
    model_name = embedding_model["name"]
    
    logger.info(f"Evaluating configuration: {config_name} with {model_name}")
    
    # Set embedding model
    os.environ["OPENAI_EMBEDDING_MODEL"] = model_name
    
    # Process document with this configuration
    document_id, chunks = process_document(config)
    
    # Add chunks to retrieval system
    logger.info(f"Adding {len(chunks)} chunks to retrieval system")
    for chunk in chunks:
        rag_agent.add_chunk(chunk)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_agent)
    
    # Find candidate chunks for each test case
    logger.info(f"Finding candidate chunks for {len(test_cases)} test cases")
    for test_case in test_cases:
        # Get top chunks
        top_chunks = rag_agent.retrieve(test_case.query, top_k=5)
        
        # Set relevant chunk IDs based on retrieval score
        # This is auto-relevance - we assume top chunks are relevant
        test_case.relevant_chunk_ids = [chunk.chunk_id for chunk in top_chunks]
    
    # Evaluate retrieval performance
    logger.info(f"Running retrieval evaluation with {len(test_cases)} test cases")
    retrieval_results = evaluator.evaluate_retrieval(test_cases)
    
    # Add configuration information to results
    retrieval_results.details["config"] = config
    retrieval_results.details["embedding_model"] = embedding_model
    
    # Save results
    results_path = EVAL_DIR / f"results_{config_name}_{model_name.replace('-', '_')}.json"
    evaluator.save_results(retrieval_results, results_path)
    
    # Print results
    logger.info(f"Results for {config_name} with {model_name}:")
    for metric, value in retrieval_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Clear retrieval system for next configuration
    rag_agent.clear()
    
    return retrieval_results

def compare_results(results_dict):
    """
    Compare results across different configurations.
    
    Args:
        results_dict: Dictionary mapping configuration names to results
    """
    logger.info("Comparing results across configurations")
    
    # Prepare comparison data
    comparison = {
        "configurations": [],
        "metrics": {}
    }
    
    # Extract metrics from all configurations
    all_metrics = set()
    for config_name, results in results_dict.items():
        all_metrics.update(results.metrics.keys())
    
    # Initialize metrics dictionary
    for metric in all_metrics:
        comparison["metrics"][metric] = {}
    
    # Populate comparison data
    for config_name, results in results_dict.items():
        # Add configuration details
        config_info = {
            "name": config_name,
            "chunking": results.details["config"],
            "embedding_model": results.details["embedding_model"]
        }
        comparison["configurations"].append(config_info)
        
        # Add metrics
        for metric in all_metrics:
            value = results.metrics.get(metric, 0)
            comparison["metrics"][metric][config_name] = value
    
    # Find best configuration for each metric
    comparison["best_configurations"] = {}
    for metric in all_metrics:
        metric_values = comparison["metrics"][metric]
        best_config = max(metric_values.items(), key=lambda x: x[1])
        comparison["best_configurations"][metric] = {
            "config": best_config[0],
            "value": best_config[1]
        }
    
    # Save comparison
    comparison_path = EVAL_DIR / "configuration_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison saved to {comparison_path}")
    
    # Print best configurations
    logger.info("Best configurations by metric:")
    for metric, best in comparison["best_configurations"].items():
        logger.info(f"  {metric}: {best['config']} ({best['value']:.4f})")
    
    return comparison

def main():
    """Run the comprehensive evaluation."""
    logger.info("Starting comprehensive evaluation")
    
    # Check if evaluation directory exists
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create test cases
    test_cases = create_test_cases()
    
    # Initialize RAG agent
    logger.info("Initializing RAG agent")
    rag_agent = RAGAgent()
    
    # Store results for comparison
    results_dict = {}
    
    # Evaluate each configuration
    for config, embedding_model in itertools.product(CHUNKING_CONFIGS, EMBEDDING_MODELS):
        config_name = f"{config['name']}_{embedding_model['name'].replace('-', '_')}"
        
        try:
            results = evaluate_configuration(config, embedding_model, rag_agent, test_cases.copy())
            results_dict[config_name] = results
        except Exception as e:
            logger.error(f"Error evaluating {config_name}: {str(e)}")
    
    # Compare results
    comparison = compare_results(results_dict)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Tested {len(CHUNKING_CONFIGS)} chunking configurations")
    logger.info(f"Tested {len(EMBEDDING_MODELS)} embedding models")
    logger.info(f"Total configurations tested: {len(results_dict)}")
    
    # Print best overall configuration
    if "precision" in comparison["best_configurations"]:
        best_precision = comparison["best_configurations"]["precision"]
        logger.info(f"\nBest configuration for precision: {best_precision['config']} ({best_precision['value']:.4f})")
    
    if "recall" in comparison["best_configurations"]:
        best_recall = comparison["best_configurations"]["recall"]
        logger.info(f"Best configuration for recall: {best_recall['config']} ({best_recall['value']:.4f})")
    
    if "f1_score" in comparison["best_configurations"]:
        best_f1 = comparison["best_configurations"]["f1_score"]
        logger.info(f"Best configuration for F1 score: {best_f1['config']} ({best_f1['value']:.4f})")
    
    logger.info("\nNext steps:")
    logger.info("1. Review the configuration comparison")
    logger.info("2. Use the best configuration for your production system")
    logger.info("3. Consider manual relevance judgments for more accurate evaluation")
    
    logger.info("Comprehensive evaluation completed")

if __name__ == "__main__":
    main() 