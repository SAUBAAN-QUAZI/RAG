#!/usr/bin/env python
"""
Evaluate RAG System Performance on LLM Book PDF
----------------------------------------------
This script processes the LLM book PDF, creates test cases,
and evaluates the RAG system's performance.
"""

import os
import sys
import logging
from pathlib import Path
import json

# Add project root to path if running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.document_processing import process_document
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

# Path to the LLM book PDF
LLM_BOOK_PATH = "data/documents/LLM book.pdf"

# Path for evaluation results
EVAL_DIR = Path("data/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def process_llm_book():
    """Process the LLM book PDF and return the document and chunks."""
    logger.info(f"Processing LLM book PDF: {LLM_BOOK_PATH}")
    
    # Check if the file exists
    if not Path(LLM_BOOK_PATH).exists():
        logger.error(f"LLM book PDF not found at {LLM_BOOK_PATH}")
        sys.exit(1)
    
    # Process the document
    results = process_document(
        LLM_BOOK_PATH,
        splitter_type="token",
        chunk_size=1000,
        chunk_overlap=200,
        save_results=True,
        title="LLM Book",
        source_type="pdf",
        category="textbook"
    )
    
    logger.info(f"Document processed. Generated {len(results['chunks'])} chunks.")
    
    # Save chunk IDs for reference
    chunk_ids = [chunk.chunk_id for chunk in results['chunks']]
    chunk_contents = {chunk.chunk_id: chunk.content for chunk in results['chunks']}
    
    with open(EVAL_DIR / "llm_book_chunks.json", "w") as f:
        json.dump({
            "total_chunks": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "chunk_contents": chunk_contents
        }, f, indent=2)
    
    logger.info(f"Chunk information saved to {EVAL_DIR / 'llm_book_chunks.json'}")
    
    return results['document'], results['chunks']

def create_test_cases():
    """Create test cases for the LLM book."""
    logger.info("Creating test cases for LLM book evaluation")
    
    # Load chunk information if available
    chunk_info_path = EVAL_DIR / "llm_book_chunks.json"
    if chunk_info_path.exists():
        with open(chunk_info_path, "r") as f:
            chunk_info = json.load(f)
        logger.info(f"Loaded information for {chunk_info['total_chunks']} chunks")
    else:
        logger.warning("Chunk information not found. Process the book first.")
        chunk_info = {"chunk_ids": [], "chunk_contents": {}}
    
    # Create test cases based on the LLM book content
    # Note: In a real scenario, you would manually identify the relevant chunk IDs
    # for each query by examining the chunk contents
    
    # For this example, we'll create test cases with placeholder chunk IDs
    # You should replace these with actual chunk IDs from your processed document
    retrieval_test_cases = [
        RetrievalTestCase(
            query="What are large language models?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            description="Basic LLM definition query"
        ),
        RetrievalTestCase(
            query="How does the transformer architecture work?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            description="Transformer architecture explanation"
        ),
        RetrievalTestCase(
            query="What is the attention mechanism?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            description="Attention mechanism explanation"
        ),
        RetrievalTestCase(
            query="What are the applications of LLMs?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            description="LLM applications query"
        ),
        RetrievalTestCase(
            query="What are the limitations of large language models?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            description="LLM limitations query"
        ),
        RetrievalTestCase(
            query="How are LLMs trained?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            description="LLM training process"
        ),
        RetrievalTestCase(
            query="What is prompt engineering?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            description="Prompt engineering query"
        ),
        RetrievalTestCase(
            query="What is fine-tuning in the context of LLMs?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            description="Fine-tuning explanation"
        ),
        RetrievalTestCase(
            query="What are embedding models?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            description="Embedding models explanation"
        ),
        RetrievalTestCase(
            query="What is the difference between GPT-3 and GPT-4?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            description="GPT model comparison"
        )
    ]
    
    # Create end-to-end test cases
    end_to_end_test_cases = [
        EndToEndTestCase(
            query="What are large language models?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            expected_answer="Large language models (LLMs) are advanced AI systems trained on vast amounts of text data that can generate human-like text, understand context, and perform various language tasks.",
            description="Basic LLM definition query"
        ),
        EndToEndTestCase(
            query="How does the transformer architecture work?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            expected_answer="The transformer architecture uses self-attention mechanisms to process input sequences in parallel, allowing it to capture long-range dependencies and relationships between words. It consists of encoder and decoder components with multiple layers of self-attention and feed-forward neural networks.",
            description="Transformer architecture explanation"
        ),
        EndToEndTestCase(
            query="What are the limitations of large language models?",
            relevant_chunk_ids=[],  # To be filled with actual chunk IDs
            expected_answer="Limitations of LLMs include hallucinations (generating false information), lack of up-to-date knowledge beyond training data, potential biases from training data, high computational requirements, and difficulty with complex reasoning tasks.",
            description="LLM limitations query"
        )
    ]
    
    # Save test cases
    test_dataset_path = EVAL_DIR / "llm_book_test_dataset.json"
    
    # Initialize RAG agent and evaluator to create the test dataset
    rag_agent = RAGAgent()
    evaluator = RAGEvaluator(rag_agent)
    
    evaluator.create_test_dataset(
        retrieval_cases=retrieval_test_cases,
        generation_cases=[],  # No generation test cases for now
        end_to_end_cases=end_to_end_test_cases,
        file_path=test_dataset_path
    )
    
    logger.info(f"Test dataset saved to {test_dataset_path}")
    
    return retrieval_test_cases, end_to_end_test_cases

def find_relevant_chunks(rag_agent, test_cases, top_k=5):
    """Find relevant chunks for each test case query."""
    logger.info("Finding relevant chunks for test queries")
    
    updated_test_cases = []
    
    for test_case in test_cases:
        logger.info(f"Processing query: {test_case.query}")
        
        # Retrieve relevant chunks
        results = rag_agent.retriever.retrieve(test_case.query, top_k=top_k)
        
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
    
    # Save updated test cases
    updated_test_dataset_path = EVAL_DIR / "llm_book_test_dataset_with_chunks.json"
    
    # Initialize evaluator to save the updated test dataset
    evaluator = RAGEvaluator(rag_agent)
    
    evaluator.create_test_dataset(
        retrieval_cases=updated_test_cases,
        generation_cases=[],
        end_to_end_cases=[],
        file_path=updated_test_dataset_path
    )
    
    logger.info(f"Updated test dataset saved to {updated_test_dataset_path}")
    
    return updated_test_cases

def evaluate_system(rag_agent, test_cases):
    """Evaluate the RAG system using the test cases."""
    logger.info("Evaluating RAG system performance")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_agent)
    
    # Evaluate retrieval performance
    logger.info("Running retrieval evaluation")
    retrieval_results = evaluator.evaluate_retrieval(test_cases)
    
    # Print results
    logger.info("Retrieval Performance:")
    for metric, value in retrieval_results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results_path = EVAL_DIR / "llm_book_retrieval_results.json"
    evaluator.save_results(retrieval_results, results_path)
    logger.info(f"Retrieval results saved to {results_path}")
    
    return retrieval_results

def main():
    """Run the LLM book evaluation."""
    logger.info("Starting LLM book evaluation")
    
    # Check if evaluation directory exists
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process the LLM book if needed
    document, chunks = process_llm_book()
    
    # Initialize RAG agent
    logger.info("Initializing RAG agent")
    rag_agent = RAGAgent()
    
    # Add document chunks to the RAG agent if not already indexed
    logger.info("Adding document chunks to the RAG agent")
    rag_agent.retriever.add_chunks(chunks)
    
    # Create test cases
    retrieval_test_cases, end_to_end_test_cases = create_test_cases()
    
    # Find relevant chunks for test cases
    updated_test_cases = find_relevant_chunks(rag_agent, retrieval_test_cases)
    
    # Evaluate system performance
    results = evaluate_system(rag_agent, updated_test_cases)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Document: {LLM_BOOK_PATH}")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Test cases: {len(updated_test_cases)}")
    logger.info("Retrieval Performance:")
    for metric, value in results.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nNext steps:")
    logger.info("1. Review the retrieved chunks for each query")
    logger.info("2. Manually identify the truly relevant chunks")
    logger.info("3. Update the test cases with the correct relevant chunk IDs")
    logger.info("4. Re-run the evaluation for a more accurate assessment")
    
    logger.info("LLM book evaluation completed")

if __name__ == "__main__":
    main() 