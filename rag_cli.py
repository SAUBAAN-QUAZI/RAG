#!/usr/bin/env python3
"""
RAG Command-Line Interface
-----------------------
This module provides a command-line interface for the RAG system.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag.document_processing.processor import process_document
from rag.retrieval.rag_agent import RAGAgent
from rag.utils import logger


def setup_argparse():
    """
    Set up the argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="RAG Command-Line Interface")
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add document command
    add_parser = subparsers.add_parser("add", help="Add a document")
    add_parser.add_argument("file_path", help="Path to the document file")
    add_parser.add_argument("--title", help="Document title")
    add_parser.add_argument("--author", help="Document author")
    add_parser.add_argument("--description", help="Document description")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--filter", help="Metadata filter in format key=value", action="append")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Start interactive mode")
    
    return parser


def parse_filters(filter_args):
    """
    Parse filter arguments into a filter dictionary.
    
    Args:
        filter_args: List of filter strings in format key=value
        
    Returns:
        dict: Filter dictionary
    """
    if not filter_args:
        return None
        
    filters = {}
    
    for filter_arg in filter_args:
        try:
            key, value = filter_arg.split("=", 1)
            filters[key.strip()] = value.strip()
        except ValueError:
            logger.warning(f"Invalid filter format: {filter_arg}. Expected format: key=value")
            
    return filters


def add_document(rag_agent, file_path, title=None, author=None, description=None):
    """
    Add a document to the RAG system.
    
    Args:
        rag_agent: RAG agent instance
        file_path: Path to the document file
        title: Document title
        author: Document author
        description: Document description
    """
    # Create metadata
    metadata = {}
    if title:
        metadata["title"] = title
    if author:
        metadata["author"] = author
    if description:
        metadata["description"] = description
        
    # Check if file exists
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return
        
    # Check file type
    if file_path.suffix.lower() != ".pdf":
        print("Error: Only PDF files are supported")
        return
        
    print(f"Processing document: {file_path}")
    
    try:
        # Add document to RAG system
        rag_agent.add_document(str(file_path), **metadata)
        print(f"Document {file_path} processed successfully")
    except Exception as e:
        print(f"Error processing document: {e}")


def query_rag(rag_agent, query, filters=None):
    """
    Query the RAG system.
    
    Args:
        rag_agent: RAG agent instance
        query: Query text
        filters: Metadata filters
    """
    print(f"Query: {query}")
    if filters:
        print(f"Filters: {filters}")
        
    try:
        # Process query
        answer = rag_agent.query(query, filter_dict=filters)
        print("\nAnswer:")
        print(answer)
    except Exception as e:
        print(f"Error processing query: {e}")


def interactive_mode(rag_agent):
    """
    Start interactive mode.
    
    Args:
        rag_agent: RAG agent instance
    """
    print("RAG Interactive Mode")
    print("--------------------")
    print("Type 'exit' or 'quit' to exit")
    print("Type 'add <file_path>' to add a document")
    print("Type 'filter key=value' to set a filter")
    print("Type 'filters' to show current filters")
    print("Type 'clear' to clear filters")
    print("Type anything else to query the RAG system")
    print()
    
    filters = {}
    
    while True:
        try:
            # Get user input
            user_input = input("> ").strip()
            
            # Check if user wants to exit
            if user_input.lower() in ("exit", "quit"):
                break
                
            # Check if user wants to add a document
            elif user_input.lower().startswith("add "):
                file_path = user_input[4:].strip()
                add_document(rag_agent, file_path)
                
            # Check if user wants to set a filter
            elif user_input.lower().startswith("filter "):
                filter_str = user_input[7:].strip()
                try:
                    key, value = filter_str.split("=", 1)
                    filters[key.strip()] = value.strip()
                    print(f"Filter set: {key.strip()}={value.strip()}")
                    print(f"Current filters: {filters}")
                except ValueError:
                    print(f"Invalid filter format: {filter_str}. Expected format: key=value")
                    
            # Check if user wants to see current filters
            elif user_input.lower() == "filters":
                print(f"Current filters: {filters}")
                
            # Check if user wants to clear filters
            elif user_input.lower() == "clear":
                filters = {}
                print("Filters cleared")
                
            # Process as a query
            elif user_input:
                query_rag(rag_agent, user_input, filters=filters if filters else None)
                
        except KeyboardInterrupt:
            break
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")
            
    print("Exiting interactive mode")


def main():
    """
    Main entry point.
    """
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it in the .env file or as an environment variable")
        sys.exit(1)
        
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Initialize RAG agent
    rag_agent = RAGAgent()
    
    # Process command
    if args.command == "add":
        add_document(
            rag_agent,
            args.file_path,
            title=args.title,
            author=args.author,
            description=args.description,
        )
    elif args.command == "query":
        filters = parse_filters(args.filter)
        query_rag(rag_agent, args.query, filters=filters)
    elif args.command == "interactive":
        interactive_mode(rag_agent)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 