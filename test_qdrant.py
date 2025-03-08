"""
Test script for Qdrant vector store integration.

This script tests connection to Qdrant Cloud and basic vector store functionality.
Run with: python test_qdrant.py
"""

import time
import uuid
import numpy as np
from pathlib import Path

# Import from our RAG system
from rag.vector_store.qdrant_store import QdrantStore
from rag.document_processing.document import DocumentChunk
from rag.config import VECTOR_DB_URL, VECTOR_DB_API_KEY
from rag.utils import logger

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

def main():
    """Test Qdrant vector store functionality."""
    
    print("\n=== Testing Qdrant Cloud Connection ===")
    print(f"Connecting to Qdrant at: {VECTOR_DB_URL}")
    
    # Use a unique collection name for testing
    test_collection = f"test_collection_{int(time.time())}"
    print(f"Using test collection: {test_collection}")
    
    try:
        # Initialize the QdrantStore with the test collection
        qdrant_store = QdrantStore(
            collection_name=test_collection,
            url=VECTOR_DB_URL,
            api_key=VECTOR_DB_API_KEY
        )
        
        print("✅ Successfully connected to Qdrant Cloud!")
        
        # Get collection statistics
        stats = qdrant_store.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Create test data
        print("\n=== Creating Test Data ===")
        
        # Create sample document chunks with pre-assigned UUIDs
        test_chunks = []
        test_embeddings = {}
        
        # Create 3 test chunks with embeddings
        for i in range(3):
            # Create a unique ID for the chunk
            chunk_id = f"test_chunk_{i}"
            
            # Create content based on the index
            if i == 0:
                content = "Qdrant is a vector database for similarity search."
                category = "databases"
            elif i == 1:
                content = "RAG systems use vector databases for effective retrieval."
                category = "rag"
            else:
                content = "Vector embeddings represent text in high-dimensional space."
                category = "embeddings"
            
            # Create the chunk
            chunk = DocumentChunk(
                content=content,
                metadata={"source": "test", "category": category},
                chunk_id=chunk_id
            )
            test_chunks.append(chunk)
            
            # Create a normalized random vector (similar to real embeddings)
            vector = np.random.rand(1536).astype(np.float32)
            vector = vector / np.linalg.norm(vector)  # Normalize to unit length
            test_embeddings[chunk_id] = vector.tolist()
        
        # Add embeddings to Qdrant
        print(f"Adding {len(test_chunks)} chunks with embeddings...")
        qdrant_store.add_embeddings(test_embeddings, test_chunks)
        print("✅ Successfully added test embeddings!")
        
        # Test search functionality
        print("\n=== Testing Search Functionality ===")
        
        # Create a test query embedding (random for testing)
        query_vector = np.random.rand(1536).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Search the collection
        print("Performing vector search...")
        results = qdrant_store.search(
            query_embedding=query_vector.tolist(),
            top_k=3
        )
        
        # Check if we got results
        if results:
            print(f"✅ Search returned {len(results)} results!")
            print("\nSample result:")
            print(f"Content: {results[0]['content'][:50]}...")
            print(f"Metadata: {results[0]['metadata']}")
            print(f"Similarity: {results[0]['similarity']}")
        else:
            print("❌ Search returned no results.")
        
        # Test filtering
        print("\n=== Testing Filtering ===")
        filtered_results = qdrant_store.search(
            query_embedding=query_vector.tolist(),
            top_k=3,
            filter_dict={"category": "rag"}
        )
        
        if filtered_results:
            print(f"✅ Filtered search returned {len(filtered_results)} results!")
            if all(result['metadata'].get('category') == 'rag' for result in filtered_results):
                print("✅ Filter was correctly applied!")
            else:
                print("❌ Filter may not be working correctly.")
        else:
            print("❌ Filtered search returned no results.")
        
        # Clean up
        print("\n=== Cleaning Up ===")
        qdrant_store.clear()
        print(f"✅ Cleared test collection: {test_collection}")
        
        print("\n=== Test Completed Successfully ===")
        print("Your Qdrant Cloud configuration is working correctly!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check that your VECTOR_DB_URL and VECTOR_DB_API_KEY are correct in .env")
        print("2. Make sure your Qdrant Cloud instance is running and accessible")
        print("3. Check if your API key has the necessary permissions")
        print("4. Verify your network connection and any firewalls")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        exit(0)
    else:
        exit(1) 