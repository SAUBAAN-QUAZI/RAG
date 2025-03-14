#!/usr/bin/env python
"""
Simple Retrieval Performance Testing
-----------------------------------
This script evaluates the performance of our enhanced semantic text splitter
compared to the default token-based splitter.

The test uses local in-memory embedding and storage to avoid external dependencies.
"""

import sys
import time
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path if running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.document_processing import process_document
from rag.document_processing.document import Document, DocumentChunk
from rag.document_processing.splitters import get_text_splitter, TextSplitter, TokenTextSplitter, SemanticTextSplitter

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
]

def process_test_document(
    file_path: str, 
    splitter_type: str,
    chunk_size: int,
    chunk_overlap: int
) -> Tuple[str, List[DocumentChunk]]:
    """
    Process the test document with the specified chunking configuration.
    
    Args:
        file_path: Path to the document
        splitter_type: Type of text splitter to use
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Number of tokens to overlap between chunks
        
    Returns:
        Tuple containing the document ID and list of chunks
    """
    logger.info(f"Processing document with splitter={splitter_type}, chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    # Process document
    result = process_document(
        file_path=file_path,
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # Extract document and chunks from result
    document = result["document"]
    chunks = result["chunks"]
    
    logger.info(f"Document {document.doc_id} processed into {len(chunks)} chunks")
    
    # Save chunks for inspection with a preview of each
    chunks_data = [
        {
            "chunk_id": chunk.metadata.get("chunk_id", str(uuid.uuid4())),
            "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            "token_count": len(chunk.content.split()),
            "char_count": len(chunk.content)
        }
        for chunk in chunks
    ]
    
    # Save chunk statistics to a file
    with open(EVAL_DIR / f"chunks_{splitter_type}.txt", "w", encoding="utf-8") as f:
        f.write(f"Document ID: {document.doc_id}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Splitter type: {splitter_type}\n")
        f.write(f"Chunk size: {chunk_size}\n")
        f.write(f"Chunk overlap: {chunk_overlap}\n\n")
        
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n")
            f.write(f"Token count: {len(chunk.content.split())}\n")
            f.write(f"Character count: {len(chunk.content)}\n")
            f.write(f"Content preview: {chunk.content[:200]}...\n\n")
    
    return document.doc_id, chunks

def analyze_chunk_quality(chunks_token: List[DocumentChunk], chunks_semantic: List[DocumentChunk]) -> Dict[str, Any]:
    """
    Compare the quality of chunks created by different splitters.
    
    Args:
        chunks_token: Chunks created by the token splitter
        chunks_semantic: Chunks created by the semantic splitter
        
    Returns:
        Dictionary of comparison metrics
    """
    # Calculate metrics for token splitter
    token_avg_length = sum(len(chunk.content) for chunk in chunks_token) / len(chunks_token)
    token_avg_tokens = sum(len(chunk.content.split()) for chunk in chunks_token) / len(chunks_token)
    token_length_variation = (
        max(len(chunk.content) for chunk in chunks_token) - 
        min(len(chunk.content) for chunk in chunks_token)
    )
    
    # Calculate metrics for semantic splitter
    semantic_avg_length = sum(len(chunk.content) for chunk in chunks_semantic) / len(chunks_semantic)
    semantic_avg_tokens = sum(len(chunk.content.split()) for chunk in chunks_semantic) / len(chunks_semantic)
    semantic_length_variation = (
        max(len(chunk.content) for chunk in chunks_semantic) - 
        min(len(chunk.content) for chunk in chunks_semantic)
    )
    
    results = {
        "token_splitter": {
            "chunk_count": len(chunks_token),
            "avg_character_length": token_avg_length,
            "avg_token_count": token_avg_tokens,
            "length_variation": token_length_variation,
        },
        "semantic_splitter": {
            "chunk_count": len(chunks_semantic),
            "avg_character_length": semantic_avg_length,
            "avg_token_count": semantic_avg_tokens,
            "length_variation": semantic_length_variation,
        }
    }
    
    return results

def main():
    """Run a simple evaluation comparing text splitters."""
    logger.info("Starting enhanced retrieval performance testing")
    
    # Ensure test document exists
    if not Path(DOCUMENT_PATH).exists():
        logger.error(f"Test document not found: {DOCUMENT_PATH}")
        return
    
    # Process document with token splitter
    logger.info("Testing token splitter")
    start_time = time.time()
    _, chunks_token = process_test_document(
        file_path=DOCUMENT_PATH,
        splitter_type="token",
        chunk_size=1000,
        chunk_overlap=200,
    )
    token_time = time.time() - start_time
    logger.info(f"Token splitter completed in {token_time:.2f} seconds, produced {len(chunks_token)} chunks")
    
    # Process document with semantic splitter
    logger.info("Testing semantic splitter")
    start_time = time.time()
    _, chunks_semantic = process_test_document(
        file_path=DOCUMENT_PATH,
        splitter_type="semantic",
        chunk_size=1500,
        chunk_overlap=300,
    )
    semantic_time = time.time() - start_time
    logger.info(f"Semantic splitter completed in {semantic_time:.2f} seconds, produced {len(chunks_semantic)} chunks")
    
    # Compare results
    logger.info("Comparing chunk quality")
    results = analyze_chunk_quality(chunks_token, chunks_semantic)
    
    # Output results
    with open(EVAL_DIR / "splitter_comparison.txt", "w", encoding="utf-8") as f:
        f.write("Splitter Comparison Results\n")
        f.write("==========================\n\n")
        
        f.write("Token Splitter:\n")
        f.write(f"Processing time: {token_time:.2f} seconds\n")
        f.write(f"Chunk count: {results['token_splitter']['chunk_count']}\n")
        f.write(f"Average characters per chunk: {results['token_splitter']['avg_character_length']:.2f}\n")
        f.write(f"Average tokens per chunk: {results['token_splitter']['avg_token_count']:.2f}\n")
        f.write(f"Length variation: {results['token_splitter']['length_variation']:.2f} characters\n\n")
        
        f.write("Semantic Splitter:\n")
        f.write(f"Processing time: {semantic_time:.2f} seconds\n")
        f.write(f"Chunk count: {results['semantic_splitter']['chunk_count']}\n")
        f.write(f"Average characters per chunk: {results['semantic_splitter']['avg_character_length']:.2f}\n")
        f.write(f"Average tokens per chunk: {results['semantic_splitter']['avg_token_count']:.2f}\n")
        f.write(f"Length variation: {results['semantic_splitter']['length_variation']:.2f} characters\n\n")
        
        # Calculate the differences
        chunk_count_diff_pct = (
            (results['semantic_splitter']['chunk_count'] - results['token_splitter']['chunk_count']) /
            results['token_splitter']['chunk_count'] * 100
        )
        
        f.write("Comparison (Semantic vs Token):\n")
        f.write(f"Chunk count difference: {chunk_count_diff_pct:.2f}%\n")
        f.write(f"Processing time difference: {(semantic_time / token_time - 1) * 100:.2f}%\n")
        
    logger.info(f"Results saved to {EVAL_DIR / 'splitter_comparison.txt'}")
    logger.info("Simple retrieval performance testing completed")

if __name__ == "__main__":
    main() 