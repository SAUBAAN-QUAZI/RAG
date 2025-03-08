#!/usr/bin/env python3
"""
RAG System Example
--------------
This script demonstrates how to use the RAG system with a simple example.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from rag.document_processing.processor import process_document
from rag.retrieval.rag_agent import RAGAgent
from rag.utils import logger


def main():
    """
    Main function to demonstrate the RAG system.
    """
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it in the .env file or as an environment variable")
        return
        
    print("RAG System Example")
    print("-----------------")
    
    # Initialize RAG agent
    rag_agent = RAGAgent()
    
    # Get PDF file path from user
    print("\nStep 1: Add a PDF document to the RAG system")
    pdf_path = input("Enter the path to a PDF file: ").strip()
    
    # Check if file exists
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: File {pdf_path} does not exist")
        return
        
    # Check file type
    if pdf_path.suffix.lower() != ".pdf":
        print("Error: Only PDF files are supported")
        return
        
    # Add document to RAG system
    print(f"\nProcessing document: {pdf_path}")
    try:
        rag_agent.add_document(str(pdf_path))
        print(f"Document {pdf_path} processed successfully")
    except Exception as e:
        print(f"Error processing document: {e}")
        return
        
    # Query the RAG system
    print("\nStep 2: Query the RAG system")
    print("You can ask questions about the document you just added.")
    print("Type 'exit' or 'quit' to exit")
    
    while True:
        # Get user query
        query = input("\nEnter your query: ").strip()
        
        # Check if user wants to exit
        if query.lower() in ("exit", "quit"):
            break
            
        # Process query
        if query:
            try:
                answer = rag_agent.query(query)
                print("\nAnswer:")
                print(answer)
            except Exception as e:
                print(f"Error processing query: {e}")
                
    print("\nThank you for trying the RAG system!")


if __name__ == "__main__":
    main() 