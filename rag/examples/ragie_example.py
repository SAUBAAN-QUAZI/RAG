#!/usr/bin/env python3
"""
Ragie.ai Integration Example
---------------------------
This script demonstrates how to use the Ragie.ai integration for document
management and RAG operations.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Add the parent directory to sys.path to import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ragie_example")

# Import our modules
from rag.integrations import create_ragie_client, RAGIE_AVAILABLE
from rag import config


def upload_document(client, file_path: str, wait_for_ready: bool = True) -> Dict[str, Any]:
    """Upload a document to Ragie and optionally wait for processing to complete"""
    
    # Check that the file exists
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return {"error": "File not found"}
    
    logger.info(f"Uploading document: {file_path}")
    
    try:
        # Prepare metadata
        metadata = {
            "source": "example_script",
            "filename": file_path.name,
        }
        
        # Upload the document
        result = client.upload_document(
            file_path=file_path,
            metadata=metadata,
            mode=config.RAGIE_PROCESS_MODE,
            partition=config.RAGIE_DEFAULT_PARTITION
        )
        
        document_id = result["id"]
        logger.info(f"Document uploaded with ID: {document_id}")
        
        # Optionally wait for the document to be ready
        if wait_for_ready:
            logger.info(f"Waiting for document {document_id} to be processed...")
            status = client.wait_for_document_ready(
                document_id=document_id,
                accept_indexed=config.RAGIE_ACCEPT_INDEXED,
                timeout=config.RAGIE_TIMEOUT,
                partition=config.RAGIE_DEFAULT_PARTITION
            )
            logger.info(f"Document processing completed with status: {status}")
            result["status"] = status
        
        return result
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return {"error": str(e)}


def list_documents(client) -> List[Dict[str, Any]]:
    """List all documents in Ragie"""
    
    logger.info("Listing all documents...")
    
    try:
        documents = client.get_all_documents(
            partition=config.RAGIE_DEFAULT_PARTITION
        )
        
        logger.info(f"Found {len(documents)} documents")
        
        # Print a summary of each document
        for i, doc in enumerate(documents):
            logger.info(f"Document {i+1}: ID={doc['id']}, Status={doc['status']}")
        
        return documents
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return []


def get_document_chunks(client, document_id: str) -> List[Dict[str, Any]]:
    """Get chunks for a specific document"""
    
    logger.info(f"Getting chunks for document {document_id}...")
    
    try:
        # Get the first page of chunks
        result = client.get_document_chunks(
            document_id=document_id,
            page_size=20,  # Get more chunks at once
            partition=config.RAGIE_DEFAULT_PARTITION
        )
        
        chunks = result["chunks"]
        
        # Handle pagination if needed
        next_cursor = result.get("next_cursor")
        pages = 1
        
        while next_cursor and pages < 5:  # Limit to 5 pages maximum
            logger.info(f"Getting next page of chunks (page {pages+1})...")
            
            result = client.get_document_chunks(
                document_id=document_id,
                page_size=20,
                cursor=next_cursor,
                partition=config.RAGIE_DEFAULT_PARTITION
            )
            
            chunks.extend(result["chunks"])
            next_cursor = result.get("next_cursor")
            pages += 1
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        
        # Print a summary of the first few chunks
        for i, chunk in enumerate(chunks[:3]):
            logger.info(f"Chunk {i+1}: ID={chunk['id']}, Index={chunk['index']}")
            # Print a preview of the text (first 50 chars)
            text_preview = chunk['text'][:50] + "..." if len(chunk['text']) > 50 else chunk['text']
            logger.info(f"  Preview: {text_preview}")
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error getting document chunks: {e}")
        return []


def get_document_content(client, document_id: str) -> Optional[str]:
    """Get the content of a document"""
    
    logger.info(f"Getting content for document {document_id}...")
    
    try:
        content = client.get_document_content(
            document_id=document_id,
            partition=config.RAGIE_DEFAULT_PARTITION
        )
        
        # Print a preview of the content
        content_preview = content[:200] + "..." if len(content) > 200 else content
        logger.info(f"Document content preview: {content_preview}")
        
        return content
    
    except Exception as e:
        logger.error(f"Error getting document content: {e}")
        return None


def get_document_summary(client, document_id: str) -> Optional[str]:
    """Get the summary of a document"""
    
    logger.info(f"Getting summary for document {document_id}...")
    
    try:
        summary = client.get_document_summary(
            document_id=document_id,
            partition=config.RAGIE_DEFAULT_PARTITION
        )
        
        logger.info(f"Document summary: {summary}")
        
        return summary
    
    except Exception as e:
        logger.error(f"Error getting document summary: {e}")
        return None


def perform_retrieval(client, query: str, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Perform a RAG retrieval operation"""
    
    logger.info(f"Performing retrieval for query: '{query}'")
    
    if document_ids:
        logger.info(f"Limiting to documents: {document_ids}")
    
    try:
        result = client.retrieve(
            query=query,
            document_ids=document_ids,
            rerank=config.ENABLE_RERANKING,
            top_k=config.TOP_K_RESULTS,
            partition=config.RAGIE_DEFAULT_PARTITION
        )
        
        chunks = result["chunks"]
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        
        # Print information about each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1}: Score={chunk['score']:.4f}, Doc={chunk['document_id']}")
            # Print a preview of the text (first 50 chars)
            text_preview = chunk['text'][:50] + "..." if len(chunk['text']) > 50 else chunk['text']
            logger.info(f"  Preview: {text_preview}")
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error performing retrieval: {e}")
        return []


def delete_document(client, document_id: str) -> bool:
    """Delete a document from Ragie"""
    
    logger.info(f"Deleting document {document_id}...")
    
    try:
        result = client.delete_document(
            document_id=document_id,
            partition=config.RAGIE_DEFAULT_PARTITION
        )
        
        logger.info(f"Document {document_id} deleted successfully")
        
        return True
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return False


def main():
    """Main function to run the example"""
    
    parser = argparse.ArgumentParser(description="Ragie.ai Integration Example")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a document")
    upload_parser.add_argument("file", help="Path to the file to upload")
    upload_parser.add_argument("--no-wait", action="store_true", help="Don't wait for processing to complete")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all documents")
    
    # Get chunks command
    chunks_parser = subparsers.add_parser("chunks", help="Get chunks for a document")
    chunks_parser.add_argument("document_id", help="ID of the document")
    
    # Get content command
    content_parser = subparsers.add_parser("content", help="Get content for a document")
    content_parser.add_argument("document_id", help="ID of the document")
    
    # Get summary command
    summary_parser = subparsers.add_parser("summary", help="Get summary for a document")
    summary_parser.add_argument("document_id", help="ID of the document")
    
    # Retrieval command
    retrieval_parser = subparsers.add_parser("retrieve", help="Perform a retrieval operation")
    retrieval_parser.add_argument("query", help="Query to retrieve chunks for")
    retrieval_parser.add_argument("--document-ids", nargs="+", help="Limit to specific document IDs")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a document")
    delete_parser.add_argument("document_id", help="ID of the document to delete")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if Ragie integration is available
    if not RAGIE_AVAILABLE:
        logger.error("Ragie integration is not available. Please install the ragie package.")
        sys.exit(1)
    
    # Check if Ragie is enabled in configuration
    if not config.USE_RAGIE:
        logger.error("Ragie integration is disabled in configuration. Set USE_RAGIE=true in .env")
        sys.exit(1)
    
    # Create Ragie client
    client = create_ragie_client(
        api_key=config.RAGIE_API_KEY,
        default_partition=config.RAGIE_DEFAULT_PARTITION
    )
    
    if client is None:
        logger.error("Failed to create Ragie client")
        sys.exit(1)
    
    # Execute the requested command
    if args.command == "upload":
        result = upload_document(client, args.file, not args.no_wait)
        if "error" in result:
            sys.exit(1)
    
    elif args.command == "list":
        list_documents(client)
    
    elif args.command == "chunks":
        get_document_chunks(client, args.document_id)
    
    elif args.command == "content":
        get_document_content(client, args.document_id)
    
    elif args.command == "summary":
        get_document_summary(client, args.document_id)
    
    elif args.command == "retrieve":
        perform_retrieval(client, args.query, args.document_ids)
    
    elif args.command == "delete":
        delete_document(client, args.document_id)
    
    else:
        logger.error("No command specified. Use --help to see available commands.")
        sys.exit(1)


if __name__ == "__main__":
    main() 