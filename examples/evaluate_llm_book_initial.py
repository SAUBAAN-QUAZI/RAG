#!/usr/bin/env python
"""
Initial Evaluation of RAG System on LLM Book PDF
-----------------------------------------------
This script runs an initial evaluation with improved chunking and
generates a template for manual relevance judgments.
"""

import sys
import logging
import json
from pathlib import Path
import time

# Add project root to path if running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.document_processing import process_document
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

# Path to the LLM book PDF
LLM_BOOK_PATH = "data/documents/LLM book.pdf"

# Path for evaluation results
EVAL_DIR = Path("data/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Improved chunking configuration
CHUNKING_CONFIG = {
    "chunk_size": 1500,  # Larger chunks for more context
    "chunk_overlap": 300,  # More overlap to avoid splitting concepts
    "splitter_type": "token"  # Token-based splitting
}

# Test queries
TEST_QUERIES = {
    "llm_definition": "What are large language models?",
    "transformer_architecture": "How does the transformer architecture work?",
    "llm_limitations": "What are the limitations of large language models?",
    "attention_mechanism": "What is the attention mechanism in transformers?",
    "llm_training": "How are large language models trained?"
}

def process_llm_book():
    """Process the LLM book PDF with improved chunking."""
    logger.info(f"Processing LLM book PDF with improved chunking")
    logger.info(f"  Chunk size: {CHUNKING_CONFIG['chunk_size']}, Overlap: {CHUNKING_CONFIG['chunk_overlap']}")
    
    # Check if the file exists
    if not Path(LLM_BOOK_PATH).exists():
        logger.error(f"LLM book PDF not found at {LLM_BOOK_PATH}")
        sys.exit(1)
    
    # Process the document
    results = process_document(
        LLM_BOOK_PATH,
        splitter_type=CHUNKING_CONFIG["splitter_type"],
        chunk_size=CHUNKING_CONFIG["chunk_size"],
        chunk_overlap=CHUNKING_CONFIG["chunk_overlap"],
        save_results=True,
        title="LLM Book",
        source_type="pdf",
        category="textbook"
    )
    
    logger.info(f"Document processed. Generated {len(results['chunks'])} chunks.")
    
    # Save chunk IDs and contents for reference
    chunk_ids = [chunk.chunk_id for chunk in results['chunks']]
    chunk_contents = {chunk.chunk_id: chunk.content for chunk in results['chunks']}
    
    output_path = EVAL_DIR / "llm_book_chunks_improved.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": CHUNKING_CONFIG,
            "total_chunks": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "chunk_contents": chunk_contents
        }, f, indent=2)
    
    logger.info(f"Chunk information saved to {output_path}")
    
    return results['document'], results['chunks']

def create_test_cases():
    """Create test cases for the LLM book."""
    logger.info("Creating test cases for LLM book evaluation")
    
    test_cases = []
    
    for query_type, query in TEST_QUERIES.items():
        # Create test case with empty relevant_chunk_ids (to be filled later)
        test_case = RetrievalTestCase(
            query=query,
            relevant_chunk_ids=[],
            description=f"{query_type} query"
        )
        test_cases.append(test_case)
        logger.info(f"Created test case: {query}")
    
    return test_cases

def find_candidate_chunks(rag_agent, test_cases, top_k=10):
    """Find candidate relevant chunks for each test case query."""
    logger.info(f"Finding candidate chunks for test queries (top_k={top_k})")
    
    candidate_chunks = {}
    
    for test_case in test_cases:
        query = test_case.query
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        start_time = time.time()
        results = rag_agent.retriever.retrieve(query, top_k=top_k)
        retrieval_time = time.time() - start_time
        
        # Extract chunk IDs
        chunk_ids = [result["chunk_id"] for result in results]
        
        # Store results
        candidate_chunks[query] = {
            "results": results,
            "retrieval_time": retrieval_time,
            "chunk_ids": chunk_ids
        }
        
        # Log the found chunks
        logger.info(f"Found {len(chunk_ids)} candidate chunks for query: {query} in {retrieval_time:.2f}s")
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
    for query, data in candidate_chunks.items():
        serializable_candidates[query] = {
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

def create_manual_relevance_template(candidate_chunks):
    """Create a template for manual relevance judgments."""
    logger.info("Creating template for manual relevance judgments")
    
    manual_relevance = {}
    
    for query, data in candidate_chunks.items():
        manual_relevance[query] = {}
        
        for result in data["results"]:
            # Default relevance score is 0 (not relevant)
            manual_relevance[query][result["chunk_id"]] = 0
    
    # Save template
    output_path = EVAL_DIR / "llm_book_manual_relevance.json"
    with open(output_path, "w") as f:
        json.dump(manual_relevance, f, indent=2)
    
    logger.info(f"Manual relevance template saved to {output_path}")
    logger.info("Please edit this file to assign relevance scores:")
    logger.info("  0 = not relevant")
    logger.info("  1 = somewhat relevant")
    logger.info("  2 = highly relevant")
    
    return manual_relevance

def evaluate_with_auto_relevance(rag_agent, test_cases):
    """Evaluate the RAG system using automatically determined relevance."""
    logger.info("Evaluating RAG system with automatically determined relevance")
    
    # Update test cases with retrieved chunks as "relevant"
    updated_test_cases = []
    
    for test_case in test_cases:
        query = test_case.query
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        results = rag_agent.retriever.retrieve(query, top_k=5)
        
        # Extract chunk IDs
        chunk_ids = [result["chunk_id"] for result in results]
        
        # Create updated test case with found chunk IDs
        updated_test_case = RetrievalTestCase(
            query=test_case.query,
            relevant_chunk_ids=chunk_ids,
            description=test_case.description
        )
        
        updated_test_cases.append(updated_test_case)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_agent)
    
    # Evaluate retrieval performance
    logger.info("Running retrieval evaluation")
    retrieval_results = evaluator.evaluate_retrieval(updated_test_cases)
    
    # Print results
    logger.info("Retrieval Performance (Auto-Relevance):")
    for metric, value in retrieval_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results_path = EVAL_DIR / "llm_book_auto_relevance_results.json"
    evaluator.save_results(retrieval_results, results_path)
    logger.info(f"Auto-relevance results saved to {results_path}")
    
    return retrieval_results

def main():
    """Run the initial LLM book evaluation."""
    logger.info("Starting initial LLM book evaluation")
    
    # Check if evaluation directory exists
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process the LLM book with improved chunking
    document, chunks = process_llm_book()
    
    # Initialize RAG agent
    logger.info("Initializing RAG agent")
    rag_agent = RAGAgent()
    
    # Add document chunks to the RAG agent
    logger.info(f"Adding {len(chunks)} chunks to retrieval system")
    rag_agent.retriever.add_chunks(chunks)
    
    # Create test cases
    test_cases = create_test_cases()
    
    # Find candidate chunks for manual review
    candidate_chunks = find_candidate_chunks(rag_agent, test_cases)
    
    # Create template for manual relevance judgments
    manual_relevance = create_manual_relevance_template(candidate_chunks)
    
    # Evaluate with automatically determined relevance
    auto_results = evaluate_with_auto_relevance(rag_agent, test_cases)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Document: {LLM_BOOK_PATH}")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Chunk size: {CHUNKING_CONFIG['chunk_size']}, Overlap: {CHUNKING_CONFIG['chunk_overlap']}")
    logger.info(f"Test cases: {len(test_cases)}")
    logger.info("Auto-Relevance Performance:")
    for metric, value in auto_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nNext steps:")
    logger.info("1. Review the candidate chunks in data/evaluation/llm_book_candidate_chunks.json")
    logger.info("2. Edit data/evaluation/llm_book_manual_relevance.json to assign relevance scores")
    logger.info("3. Run the full evaluation script with manual relevance judgments")
    
    logger.info("Initial LLM book evaluation completed")

if __name__ == "__main__":
    main() 