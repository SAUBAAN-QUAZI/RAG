# test_qdrant_connection.py
import os
import logging
from dotenv import load_dotenv
from rag.vector_store.qdrant_store import QdrantStore
from rag.utils import logger

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

def test_qdrant():
    """Test the connection to Qdrant."""
    logger.info("Testing Qdrant connection...")
    
    # Create QdrantStore
    qdrant = QdrantStore(collection_name="test_connection")
    
    # Test getting collections
    logger.info("Getting collections...")
    result = qdrant._make_request('GET', 'collections')
    logger.info(f"Collections response: {result}")
    
    logger.info("Qdrant connection test completed.")

if __name__ == "__main__":
    test_qdrant()