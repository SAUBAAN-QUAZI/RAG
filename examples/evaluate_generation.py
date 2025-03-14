#!/usr/bin/env python
"""
Evaluation of RAG System Generation Quality
------------------------------------------
This script evaluates the generation quality of the RAG system using
a set of test cases with expected answers.
"""

import sys
import logging
import json
import time
from pathlib import Path
import os

# Add project root to path if running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retrieval.rag_agent import RAGAgent
from rag.evaluation import (
    RAGEvaluator,
    GenerationTestCase,
    EndToEndTestCase
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

# Generation test cases with expected answers
GENERATION_TEST_CASES = [
    {
        "query": "What are large language models?",
        "expected_answer": "Large language models (LLMs) are advanced AI systems trained on vast amounts of text data to understand and generate human-like text. They use neural network architectures, typically based on transformers, to process and generate language. LLMs can perform various tasks such as text completion, translation, summarization, and question answering without task-specific training."
    },
    {
        "query": "How does the transformer architecture work?",
        "expected_answer": "The transformer architecture works through a self-attention mechanism that allows the model to weigh the importance of different words in a sequence when processing language. It consists of an encoder-decoder structure with multiple layers of self-attention and feed-forward neural networks. The key innovation is the attention mechanism that enables the model to focus on relevant parts of the input when generating each word of the output, allowing for more effective processing of long-range dependencies in text."
    },
    {
        "query": "What are the limitations of large language models?",
        "expected_answer": "Large language models have several limitations: they can generate factually incorrect information (hallucinations), lack true understanding of the content they process, have limited reasoning capabilities, can perpetuate biases present in training data, have a knowledge cutoff date, struggle with complex mathematical reasoning, lack real-time information, consume significant computational resources, and may have difficulty with specialized domain knowledge unless specifically fine-tuned."
    },
    {
        "query": "How can hallucinations in large language models be reduced?",
        "expected_answer": "Hallucinations in large language models can be reduced through several techniques: retrieval-augmented generation (RAG) to ground responses in verified information, fine-tuning on high-quality data, implementing fact-checking mechanisms, using self-consistency techniques to cross-check generated content, applying structured prompting strategies, implementing confidence scoring, and using human feedback to improve model outputs over time."
    }
]

def create_generation_test_cases():
    """
    Create generation test cases from the predefined list.
    
    Returns:
        List of generation test cases
    """
    logger.info("Creating generation test cases")
    
    test_cases = []
    
    for case in GENERATION_TEST_CASES:
        test_case = GenerationTestCase(
            query=case["query"],
            context="",  # Will be populated during evaluation
            expected_answer=case["expected_answer"],
            description=f"Generation test case"
        )
        test_cases.append(test_case)
        logger.info(f"Created test case: {case['query']}")
    
    return test_cases

def create_end_to_end_test_cases():
    """
    Create end-to-end test cases from the predefined list.
    
    Returns:
        List of end-to-end test cases
    """
    logger.info("Creating end-to-end test cases")
    
    test_cases = []
    
    for case in GENERATION_TEST_CASES:
        test_case = EndToEndTestCase(
            query=case["query"],
            relevant_chunk_ids=[],  # Will be populated during evaluation
            expected_answer=case["expected_answer"],
            description=f"End-to-end test case"
        )
        test_cases.append(test_case)
        logger.info(f"Created test case: {case['query']}")
    
    return test_cases

def evaluate_generation(rag_agent, test_cases):
    """
    Evaluate generation quality using the test cases.
    
    Args:
        rag_agent: The RAG agent to use for evaluation
        test_cases: List of generation test cases
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating generation quality with {len(test_cases)} test cases")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_agent)
    
    # Create new test cases with populated context
    updated_test_cases = []
    
    for test_case in test_cases:
        # Retrieve relevant chunks
        chunks = rag_agent.retriever.retrieve(test_case.query, top_k=3)
        
        # Create context from chunks
        context = "\n\n".join([chunk["content"] for chunk in chunks])
        
        # Create a new test case with the context
        updated_test_case = GenerationTestCase(
            query=test_case.query,
            context=context,
            expected_answer=test_case.expected_answer,
            description=test_case.description
        )
        
        updated_test_cases.append(updated_test_case)
    
    # Evaluate generation performance
    logger.info("Running generation evaluation")
    generation_results = evaluator.evaluate_generation(updated_test_cases)
    
    # Print results
    logger.info("Generation Performance:")
    for metric, value in generation_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results_path = EVAL_DIR / "generation_results.json"
    evaluator.save_results(generation_results, results_path)
    logger.info(f"Generation results saved to {results_path}")
    
    return generation_results

def evaluate_end_to_end(rag_agent, test_cases):
    """
    Evaluate end-to-end performance using the test cases.
    
    Args:
        rag_agent: The RAG agent to use for evaluation
        test_cases: List of end-to-end test cases
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating end-to-end performance with {len(test_cases)} test cases")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_agent)
    
    # Create new test cases with populated relevant_chunk_ids
    updated_test_cases = []
    
    for test_case in test_cases:
        # Retrieve relevant chunks
        chunks = rag_agent.retriever.retrieve(test_case.query, top_k=3)
        
        # Extract chunk IDs
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        
        # Create a new test case with the chunk IDs
        updated_test_case = EndToEndTestCase(
            query=test_case.query,
            relevant_chunk_ids=chunk_ids,
            expected_answer=test_case.expected_answer,
            metadata_filters=test_case.metadata_filters,
            description=test_case.description
        )
        
        updated_test_cases.append(updated_test_case)
    
    # Evaluate end-to-end performance
    logger.info("Running end-to-end evaluation")
    end_to_end_results = evaluator.evaluate_end_to_end(updated_test_cases)
    
    # Print results
    logger.info("End-to-End Performance:")
    for metric, value in end_to_end_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results_path = EVAL_DIR / "end_to_end_results.json"
    evaluator.save_results(end_to_end_results, results_path)
    logger.info(f"End-to-end results saved to {results_path}")
    
    return end_to_end_results

def analyze_generation_examples(generation_results):
    """
    Analyze generation examples to identify patterns and issues.
    
    Args:
        generation_results: Generation evaluation results
    """
    logger.info("Analyzing generation examples")
    
    # Extract test case results
    test_case_results = generation_results.details.get("test_case_results", [])
    
    # Prepare analysis
    analysis = {
        "high_quality_examples": [],
        "low_quality_examples": [],
        "common_issues": {},
        "summary": {}
    }
    
    # Analyze each test case
    for result in test_case_results:
        query = result.get("query", "")
        expected = result.get("expected_answer", "")
        generated = result.get("generated_answer", "")
        metrics = result.get("metrics", {})
        
        # Determine if high or low quality based on factual consistency
        factual_consistency = metrics.get("factual_consistency", 0)
        
        example = {
            "query": query,
            "expected_answer": expected,
            "generated_answer": generated,
            "metrics": metrics
        }
        
        if factual_consistency >= 0.8:
            analysis["high_quality_examples"].append(example)
        else:
            analysis["low_quality_examples"].append(example)
            
            # Identify potential issues
            if factual_consistency < 0.5:
                issue = "low_factual_consistency"
                analysis["common_issues"][issue] = analysis["common_issues"].get(issue, 0) + 1
            
            if metrics.get("answer_relevance", 1) < 0.7:
                issue = "low_relevance"
                analysis["common_issues"][issue] = analysis["common_issues"].get(issue, 0) + 1
    
    # Prepare summary
    analysis["summary"] = {
        "total_examples": len(test_case_results),
        "high_quality_count": len(analysis["high_quality_examples"]),
        "low_quality_count": len(analysis["low_quality_examples"]),
        "high_quality_percentage": len(analysis["high_quality_examples"]) / len(test_case_results) * 100 if test_case_results else 0,
        "common_issues": analysis["common_issues"]
    }
    
    # Save analysis
    analysis_path = EVAL_DIR / "generation_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Generation analysis saved to {analysis_path}")
    
    # Print summary
    logger.info("Generation Analysis Summary:")
    logger.info(f"  Total examples: {analysis['summary']['total_examples']}")
    logger.info(f"  High quality examples: {analysis['summary']['high_quality_count']} ({analysis['summary']['high_quality_percentage']:.1f}%)")
    logger.info(f"  Low quality examples: {analysis['summary']['low_quality_count']}")
    
    if analysis["common_issues"]:
        logger.info("  Common issues:")
        for issue, count in analysis["common_issues"].items():
            logger.info(f"    {issue}: {count}")
    
    return analysis

def main():
    """Run the generation evaluation."""
    logger.info("Starting generation evaluation")
    
    # Check if evaluation directory exists
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize RAG agent
    logger.info("Initializing RAG agent")
    rag_agent = RAGAgent()
    
    # Check if there are chunks in the system
    # Modified to check if the retriever has a vector store with data
    try:
        # Try to get collection stats to see if there are vectors
        stats = rag_agent.retriever.vector_store.get_collection_stats()
        has_chunks = stats.get("vectors_count", 0) > 0
        if not has_chunks:
            logger.warning("No chunks found in the retrieval system")
            logger.warning("Please run document processing first to add chunks to the system")
            logger.info("Continuing with evaluation, but results may not be meaningful")
    except Exception as e:
        logger.warning(f"Could not check for chunks: {e}")
        logger.warning("Continuing with evaluation, but results may not be meaningful")
    
    # Create test cases
    generation_test_cases = create_generation_test_cases()
    end_to_end_test_cases = create_end_to_end_test_cases()
    
    # Evaluate generation quality
    generation_results = evaluate_generation(rag_agent, generation_test_cases)
    
    # Evaluate end-to-end performance
    end_to_end_results = evaluate_end_to_end(rag_agent, end_to_end_test_cases)
    
    # Analyze generation examples
    analysis = analyze_generation_examples(generation_results)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    logger.info("Generation Performance:")
    for metric, value in generation_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nEnd-to-End Performance:")
    for metric, value in end_to_end_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nNext steps:")
    logger.info("1. Review the generation analysis to identify patterns and issues")
    logger.info("2. Improve the retrieval system to provide better context")
    logger.info("3. Adjust generation parameters for better quality")
    logger.info("4. Add more test cases for comprehensive evaluation")
    
    logger.info("Generation evaluation completed")

if __name__ == "__main__":
    main() 