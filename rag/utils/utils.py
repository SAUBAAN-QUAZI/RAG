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