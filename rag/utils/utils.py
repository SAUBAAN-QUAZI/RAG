"""
Utility Module
-------------
Common utility functions used across the RAG system.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag")


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Generate a SHA-256 hash of a file's contents.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Hexadecimal hash of the file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read and update hash in chunks for memory efficiency
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()


def get_text_hash(text: str) -> str:
    """
    Generate a SHA-256 hash of a text string.
    
    Args:
        text: Text to hash
        
    Returns:
        str: Hexadecimal hash of the text
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Data saved to {file_path}")


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        The loaded data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return data


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    max_retries: int = 5,
    errors: tuple = (Exception,),
):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: The function to execute
        initial_delay: Initial delay between retries in seconds
        exponential_base: Base of the exponential to use for backoff
        max_retries: Maximum number of retries
        errors: Tuple of exceptions to catch and retry
        
    Returns:
        A wrapped function that will be retried with exponential backoff
    """
    def wrapper(*args, **kwargs):
        delay = initial_delay
        
        for retry in range(max_retries):
            try:
                return func(*args, **kwargs)
            except errors as e:
                if retry == max_retries - 1:
                    raise
                
                logger.warning(
                    f"Retrying {func.__name__} in {delay} seconds due to {e}"
                )
                time.sleep(delay)
                delay *= exponential_base
                
    return wrapper 


def dump_debug_info(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Dump debugging information to a file in a readable format.
    
    Args:
        data: Data to dump
        file_path: Path to save the data
        indent: JSON indentation
    """
    # Ensure the directory exists
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to handle different data types appropriately
    try:
        if isinstance(data, dict):
            # For dictionaries, write as JSON
            with open(file_path, "w") as f:
                json.dump(data, f, indent=indent, default=str)
        elif isinstance(data, (list, tuple)):
            # For lists or tuples, format each item and write as JSON
            with open(file_path, "w") as f:
                json.dump(data, f, indent=indent, default=str)
        else:
            # For other types, use string representation
            with open(file_path, "w") as f:
                f.write(str(data))
                
        logger.info(f"Debug information saved to {file_path}")
    except Exception as e:
        logger.error(f"Error dumping debug information: {e}")
        

def debug_chunk_embeddings(chunks: List, embeddings: Dict, output_dir: Union[str, Path]) -> None:
    """
    Save detailed debug information about chunks and embeddings.
    
    Args:
        chunks: List of document chunks
        embeddings: Dictionary mapping chunk IDs to embeddings
        output_dir: Directory to save debug files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a summary of chunk information
    chunk_summary = []
    for i, chunk in enumerate(chunks):
        chunk_summary.append({
            "index": i,
            "chunk_id": chunk.chunk_id,
            "content_preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
            "has_embedding": str(chunk.chunk_id) in embeddings,
            "metadata_keys": list(chunk.metadata.keys()) if hasattr(chunk, "metadata") else []
        })
    
    # Save chunk summary
    dump_debug_info(chunk_summary, output_dir / "chunk_summary.json")
    
    # Save embedding keys and metadata
    embedding_info = {
        "total_embeddings": len(embeddings),
        "embedding_ids": list(embeddings.keys()),
        "embedding_dimensions": len(next(iter(embeddings.values()))) if embeddings else 0,
        "embedding_statistics": {
            "min": min(len(emb) for emb in embeddings.values()) if embeddings else 0,
            "max": max(len(emb) for emb in embeddings.values()) if embeddings else 0
        }
    }
    dump_debug_info(embedding_info, output_dir / "embedding_info.json")
    
    # Compare IDs and identify mismatches
    chunk_ids = [str(c.chunk_id) for c in chunks]
    embedding_ids = list(embeddings.keys())
    
    id_analysis = {
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "matching_ids": len(set(chunk_ids).intersection(set(embedding_ids))),
        "missing_from_embeddings": list(set(chunk_ids) - set(embedding_ids)),
        "extra_in_embeddings": list(set(embedding_ids) - set(chunk_ids)),
    }
    dump_debug_info(id_analysis, output_dir / "id_analysis.json")
    
    logger.info(f"Detailed debug information saved to {output_dir}") 