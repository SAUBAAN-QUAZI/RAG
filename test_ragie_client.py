#!/usr/bin/env python
"""
Test script for the Ragie integration

This script tests the main functionalities of the RagieClient class
to ensure it works properly with Ragie SDK v1.5.0.
"""

import logging
import sys
import json
import os
from pathlib import Path
from rag.integrations.ragie import RagieClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_document_listing(client):
    """Test listing all documents"""
    print("\n=== Testing Document Listing ===")
    try:
        docs = client.get_all_documents()
        print(f"Found {len(docs)} documents")
        
        if docs:
            print(f"\nListing first {min(3, len(docs))} documents:")
            for i, doc in enumerate(docs[:3]):
                print(f"Document {i+1}: {doc}")
            
            # Create a test JSON file for the frontend to use
            print("\nCreating a test document list JSON file...")
            with open("test_documents.json", "w") as f:
                json.dump({
                    "documents": docs,
                    "pagination": {
                        "page": 1,
                        "page_size": len(docs),
                        "total_pages": 1,
                        "total_documents": len(docs)
                    }
                }, f, indent=2)
            print("Created test_documents.json with document list")
        return docs
    except Exception as e:
        logger.error(f"Error in document listing: {e}")
        print(f"❌ Document listing failed: {e}")
        return []

def test_document_status(client, docs):
    """Test getting document status"""
    print("\n=== Testing Document Status ===")
    
    # Test document status if we have valid document IDs
    valid_docs = [doc for doc in docs if doc["id"] and doc["id"] != "unknown"]
    if valid_docs:
        doc_id = valid_docs[0]["id"]
        print(f"Testing get_document_status with document {doc_id}...")
        try:
            status = client.get_document_status(doc_id)
            print(f"✅ Status: {status}")
            return True
        except Exception as e:
            print(f"❌ Error getting document status: {e}")
            return False
    else:
        print("⚠️ No valid document IDs found for status testing")
        return False

def test_document_retrieval(client, query="test"):
    """Test document retrieval"""
    print("\n=== Testing Document Retrieval ===")
    try:
        print(f"Testing retrieval with query: '{query}'")
        results = client.retrieve(query=query, top_k=3)
        
        if "chunks" in results:
            chunks = results["chunks"]
            print(f"✅ Retrieved {len(chunks)} chunks")
            
            if chunks:
                print("\nSample chunk:")
                sample = chunks[0]
                print(f"Text (first 100 chars): {sample['text'][:100]}...")
                print(f"Score: {sample['score']}")
                print(f"Document ID: {sample['document_id']}")
            return True
        else:
            print("❌ No chunks found in retrieval results")
            return False
    except Exception as e:
        print(f"❌ Error in document retrieval: {e}")
        return False

def test_document_upload(client, test_file_path=None):
    """Test document upload if a test file is provided"""
    print("\n=== Testing Document Upload ===")
    
    if not test_file_path:
        # Check if we have a sample PDF for testing
        sample_paths = [
            "sample.pdf", 
            "test.pdf",
            "data/sample.pdf", 
            "tests/sample.pdf"
        ]
        
        for path in sample_paths:
            if Path(path).exists():
                test_file_path = path
                break
    
    if not test_file_path or not Path(test_file_path).exists():
        print("⚠️ No test file found for upload testing")
        return False
    
    try:
        print(f"Uploading test document: {test_file_path}")
        result = client.upload_document(
            file_path=test_file_path,
            metadata={"test": True, "purpose": "integration_test"}
        )
        
        print("✅ Document uploaded successfully")
        print(f"Document ID: {result['id']}")
        print(f"Status: {result['status']}")
        
        # Test waiting for document to be ready
        try:
            print("\nWaiting for document to be ready (max 30 seconds)...")
            status = client.wait_for_document_ready(
                document_id=result['id'],
                timeout=30,
                interval=5
            )
            print(f"✅ Document ready with status: {status}")
        except TimeoutError:
            print("⚠️ Document processing timeout - but upload was successful")
        
        return True
    except Exception as e:
        print(f"❌ Error uploading document: {e}")
        return False

def main():
    """Test the Ragie client integration"""
    try:
        print("Initializing RagieClient...")
        client = RagieClient()
        print("✅ RagieClient initialized successfully")
        
        # Test document listing
        docs = test_document_listing(client)
        
        # Test document status
        if docs:
            test_document_status(client, docs)
        
        # Test document retrieval
        test_document_retrieval(client)
        
        # Test document upload (if a test file is provided via environment variable)
        test_file = os.environ.get("TEST_PDF_PATH")
        if test_file:
            test_document_upload(client, test_file)
        
        print("\n✅ All tests completed!")
        return 0
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        print(f"\n❌ Testing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 