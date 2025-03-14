#!/usr/bin/env python
"""
Debug Embedding IDs
------------------
This script helps diagnose chunk ID mapping issues by processing a test document
and tracking IDs throughout the embedding and storage pipeline.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path if running script directly
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.insert(0, str(project_root))

# Enable debug mode
os.environ["RAG_DEBUG"] = "true"

from rag.config import EMBEDDING_MODEL
from rag.document_processing.processor import process_document
from rag.embedding.service import EmbeddingService
from rag.vector_store.qdrant_store import QdrantStore
from rag.retrieval.retriever import Retriever
from rag.utils import logger
from rag.utils.utils import dump_debug_info, debug_chunk_embeddings


def test_document_processing(file_path):
    """Process a test document and track IDs."""
    logger.info(f"Testing document processing with file: {file_path}")
    logger.info(f"Using embedding model: {EMBEDDING_MODEL}")
    
    # Create debug directory
    debug_dir = Path("data/debug") / f"id_debug_{int(time.time())}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Process document to generate chunks
    logger.info("Processing document...")
    result = process_document(file_path, save_results=True)
    
    document = result["document"]
    chunks = result["chunks"]
    
    logger.info(f"Document processed: {document.doc_id}, generated {len(chunks)} chunks")
    
    # Save document and chunk IDs for reference
    doc_info = {
        "doc_id": document.doc_id,
        "chunk_count": len(chunks),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "chunk_id_sample": [chunk.chunk_id for chunk in chunks[:5]],
    }
    dump_debug_info(doc_info, debug_dir / "document_info.json")
    
    # Generate embeddings directly
    logger.info("Generating embeddings...")
    embedding_service = EmbeddingService(use_cache=False)
    embeddings = embedding_service.embed_chunks(chunks)
    
    # Debug embeddings
    logger.info("Analyzing embeddings...")
    debug_chunk_embeddings(chunks, embeddings, debug_dir / "embeddings")
    
    # Test Qdrant store directly
    logger.info("Testing Qdrant store...")
    test_collection = f"test_collection_{int(time.time())}"
    qdrant = QdrantStore(collection_name=test_collection)
    
    # Save Qdrant initial state
    try:
        qdrant_info = {
            "collection_name": test_collection,
            "vector_size": qdrant.vector_size,
            "embedding_model": EMBEDDING_MODEL,
        }
        dump_debug_info(qdrant_info, debug_dir / "qdrant_info.json")
    except Exception as e:
        logger.error(f"Error getting Qdrant info: {e}")
    
    # Add embeddings to Qdrant
    logger.info(f"Adding {len(embeddings)} embeddings to Qdrant...")
    qdrant.add_embeddings(embeddings, chunks)
    
    # Test retrieval with a sample query
    logger.info("Testing retrieval...")
    retriever = Retriever(embedding_service=embedding_service, 
                         vector_store_type="qdrant",
                         collection_name=test_collection)
    
    sample_query = "What is this document about?"
    logger.info(f"Sample query: {sample_query}")
    
    results = retriever.retrieve(sample_query, top_k=3)
    
    # Save retrieval results
    retrieval_info = {
        "query": sample_query,
        "result_count": len(results),
        "results": results,
    }
    dump_debug_info(retrieval_info, debug_dir / "retrieval_results.json")
    
    logger.info(f"Debug information saved to {debug_dir}")
    return debug_dir


def main():
    """Main function to run the debug script."""
    print("Debug Embedding ID Mapping Script")
    print("-------------------------------")
    print("This script will process a test document and track chunk IDs")
    print("throughout the embedding and storage pipeline.")
    print()
    
    # Get test document path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter the path to a test document (PDF): ").strip()
    
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist")
        return
    
    # Process test document
    debug_dir = test_document_processing(file_path)
    
    print()
    print(f"Debug complete! All information saved to: {debug_dir}")
    print()
    print("To analyze the results:")
    print(f"1. Check the document_info.json file for document and chunk IDs")
    print(f"2. Check the embeddings directory for detailed embedding analysis")
    print(f"3. Check the retrieval_results.json file for search results")
    

if __name__ == "__main__":
    main() 