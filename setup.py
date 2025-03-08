#!/usr/bin/env python3
"""
RAG System Setup
------------
This script helps set up the RAG system environment.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv not installed")
    print("Please install it with pip install python-dotenv")
    sys.exit(1)


def create_dirs():
    """
    Create necessary directories.
    """
    print("Creating directories...")
    
    # Get base directory
    base_dir = Path(__file__).resolve().parent
    
    # Create directories
    dirs = [
        base_dir / "data",
        base_dir / "data/documents",
        base_dir / "data/chunks",
        base_dir / "data/vectors",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}")
        
    print("Directories created successfully")


def check_env_file():
    """
    Check if .env file exists and has required variables.
    """
    print("Checking environment file...")
    
    # Get base directory
    base_dir = Path(__file__).resolve().parent
    env_file = base_dir / ".env"
    env_example_file = base_dir / ".env.example"
    
    # Check if .env file exists
    if not env_file.exists():
        print("  .env file not found")
        
        # Check if .env.example file exists
        if env_example_file.exists():
            print("  Creating .env file from .env.example")
            
            # Copy .env.example to .env
            with open(env_example_file, "r") as src:
                with open(env_file, "w") as dst:
                    dst.write(src.read())
                    
            print("  .env file created successfully")
            print("  Please edit the .env file and set your OpenAI API key")
        else:
            print("  .env.example file not found")
            print("  Creating basic .env file")
            
            # Create basic .env file
            with open(env_file, "w") as f:
                f.write("# OpenAI API credentials\n")
                f.write("OPENAI_API_KEY=your_openai_api_key\n")
                
            print("  Basic .env file created successfully")
            print("  Please edit the .env file and set your OpenAI API key")
    else:
        print("  .env file found")
        
        # Load environment variables
        load_dotenv(env_file)
        
        # Check if OpenAI API key is set
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or openai_api_key == "your_openai_api_key":
            print("  Warning: OPENAI_API_KEY not set in .env file")
            print("  Please edit the .env file and set your OpenAI API key")
        else:
            print("  OPENAI_API_KEY found in .env file")


def main():
    """
    Main function to set up the RAG system environment.
    """
    print("RAG System Setup")
    print("---------------")
    
    # Create directories
    create_dirs()
    print()
    
    # Check environment file
    check_env_file()
    print()
    
    print("Setup completed successfully")
    print()
    print("Next steps:")
    print("1. Make sure your OpenAI API key is set in the .env file")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Try the example: python example.py")
    print("4. Start the API server: python -m uvicorn app.main:app --reload")
    print("5. Or use the CLI: python rag_cli.py interactive")


if __name__ == "__main__":
    main() 