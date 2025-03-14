#!/usr/bin/env python
"""
Reset Vector Database
--------------------
This script deletes and recreates the vector database collection with
the correct dimensions for the currently configured embedding model.
"""

import os
import sys
from pathlib import Path

# Add project root to path if running script directly
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.insert(0, str(project_root))

from rag.config import VECTOR_DB_TYPE, EMBEDDING_MODEL
from rag.vector_store.qdrant_store import QdrantStore
from rag.utils import logger


def reset_qdrant_collection(collection_name="rag_collection", confirm=True):
    """
    Reset (delete and recreate) a Qdrant collection.
    
    Args:
        collection_name: Name of the collection to reset
        confirm: Whether to ask for confirmation before deleting
    """
    logger.info(f"Preparing to reset Qdrant collection: {collection_name}")
    logger.info(f"Current embedding model: {EMBEDDING_MODEL}")
    
    # Determine vector size based on model
    vector_size = 3072 if "large" in EMBEDDING_MODEL else 1536
    logger.info(f"Using vector size: {vector_size} dimensions")
    
    # Ask for confirmation
    if confirm:
        response = input(f"This will DELETE the existing collection '{collection_name}' and recreate it. Continue? (y/n): ")
        if response.lower() != 'y':
            logger.info("Operation cancelled")
            return
    
    try:
        # Create QdrantStore instance
        qdrant = QdrantStore(collection_name=collection_name)
        
        # Delete collection
        logger.info(f"Deleting collection: {collection_name}")
        qdrant.clear()
        logger.info(f"Collection {collection_name} deleted")
        
        # Force recreation of collection with correct vector size
        logger.info(f"Recreating collection with {vector_size} dimensions")
        qdrant._create_collection_if_not_exists()
        logger.info(f"Collection {collection_name} recreated successfully")
        
        # Verify collection exists and has correct dimensions
        try:
            stats = qdrant.get_stats()
            logger.info(f"Collection stats: {stats}")
            logger.info("Reset completed successfully")
        except Exception as e:
            logger.error(f"Error verifying collection: {e}")
    
    except Exception as e:
        logger.error(f"Error resetting collection: {e}")
        raise


def main():
    """
    Main function to reset vector database.
    """
    print("Reset Vector Database Utility")
    print("-----------------------------")
    print(f"Current vector database type: {VECTOR_DB_TYPE}")
    print(f"Current embedding model: {EMBEDDING_MODEL}")
    print()
    
    if VECTOR_DB_TYPE.lower() != "qdrant":
        print(f"This script only supports Qdrant. Current setting: {VECTOR_DB_TYPE}")
        return
    
    # Reset with default collection name
    reset_qdrant_collection()
    
    print("\nTo reindex your documents after resetting the vector database:")
    print("1. Delete the existing chunk index files if needed")
    print("2. Process your documents again using the document processor")
    print("3. Or simply re-upload your documents through the API")


if __name__ == "__main__":
    main() 