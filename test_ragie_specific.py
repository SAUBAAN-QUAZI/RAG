#!/usr/bin/env python
"""
Specific test script for Ragie document listing and source retrieval
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

def test_list_documents(client):
    """Test the list_documents API endpoint with pagination and page_size"""
    print("\n=== Testing List Documents API ===")
    try:
        # Test with default page size (should be 10)
        print("Testing with default page size...")
        docs = client.get_all_documents()
        print(f"Retrieved {len(docs)} documents with default page size")
        
        # Display document IDs
        print("\nDocument IDs found in the system:")
        for i, doc in enumerate(docs):
            print(f"Document {i+1}: ID={doc.get('id', 'unknown')}, Status={doc.get('status', 'unknown')}")
        
        # Return the document list for further testing
        return docs
    except Exception as e:
        logger.exception(f"Error listing documents: {e}")
        print(f"❌ Document listing failed: {e}")
        return []

def test_get_document_source(client, document_id):
    """Test retrieving a document's source file"""
    print(f"\n=== Testing Get Document Source API for {document_id} ===")
    try:
        # Try to retrieve the document source
        print(f"Retrieving source for document ID: {document_id}")
        source_bytes = client.get_document_source(document_id)
        
        # Write to a temp file so we can see it worked
        output_path = f"document_{document_id}_source.bin"
        with open(output_path, "wb") as f:
            f.write(source_bytes)
        
        print(f"✅ Successfully retrieved document source ({len(source_bytes)} bytes)")
        print(f"Source saved to: {output_path}")
        return True
    except Exception as e:
        logger.exception(f"Error retrieving document source: {e}")
        print(f"❌ Document source retrieval failed: {e}")
        return False

def main():
    """Run specific tests for document listing and source retrieval"""
    try:
        print("Initializing RagieClient...")
        client = RagieClient()
        print("✅ RagieClient initialized successfully")
        
        # Test document listing
        docs = test_list_documents(client)
        
        # If we have documents, test source retrieval on the first one
        if docs:
            valid_docs = [doc for doc in docs if doc.get("id") and doc.get("id") != "unknown"]
            if valid_docs:
                doc_id = valid_docs[0]["id"]
                test_get_document_source(client, doc_id)
            else:
                print("\n⚠️ No valid document IDs found for source testing")
        
        # Test with a specific document ID if provided
        specific_id = os.environ.get("DOCUMENT_ID")
        if specific_id:
            print(f"\nTesting with specific document ID from environment: {specific_id}")
            test_get_document_source(client, specific_id)
        
        print("\n✅ All specific tests completed!")
        return 0
    except Exception as e:
        logger.exception(f"Error during testing: {e}")
        print(f"\n❌ Testing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 