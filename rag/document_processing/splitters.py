"""
Text Splitters
-----------
This module contains text splitters for dividing documents into chunks.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import tiktoken

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP
from rag.document_processing.document import Document, DocumentChunk
from rag.utils import logger


class TextSplitter(ABC):
    """
    Abstract base class for text splitters.
    """
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: The text to split
            
        Returns:
            List[str]: List of text chunks
        """
        pass
    
    def split_document(self, document: Document) -> List[DocumentChunk]:
        """
        Split a document into chunks.
        
        Args:
            document: The document to split
            
        Returns:
            List[DocumentChunk]: List of document chunks
        """
        # Split the document text into chunks
        text_chunks = self.split_text(document.content)
        
        # Create document chunks
        doc_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Create metadata for the chunk
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "doc_id": document.doc_id,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
            })
            
            # Create document chunk
            chunk = DocumentChunk(content=chunk_text, metadata=chunk_metadata)
            doc_chunks.append(chunk)
            
        return doc_chunks


class TokenTextSplitter(TextSplitter):
    """
    Split text into chunks based on token count.
    
    This splitter uses the tiktoken library to count tokens according to
    a specific tokenizer (usually matching the one used by the LLM).
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        encoding_name: str = "cl100k_base",  # Default for GPT-4, GPT-3.5-Turbo
    ):
        """
        Initialize a TokenTextSplitter.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            encoding_name: Name of the tiktoken encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        
        # Load the tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
            
        logger.info(f"Initialized TokenTextSplitter with chunk_size={chunk_size}, "
                   f"chunk_overlap={chunk_overlap}, encoding={encoding_name}")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: The text to split
            
        Returns:
            List[str]: List of text chunks
        """
        if not text:
            return []
            
        # Encode the text
        tokens = self.tokenizer.encode(text)
        
        # Split tokens into chunks
        chunks = []
        i = 0
        while i < len(tokens):
            # Get chunk tokens
            chunk_end = min(i + self.chunk_size, len(tokens))
            chunk_tokens = tokens[i:chunk_end]
            
            # Decode chunk
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move to next chunk, considering overlap
            i += self.chunk_size - self.chunk_overlap
            
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks


class SentenceTextSplitter(TextSplitter):
    """
    Split text into chunks based on sentences.
    
    This splitter tries to keep sentences together when creating chunks,
    only breaking at sentence boundaries when possible.
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        encoding_name: str = "cl100k_base",
        separator: str = ".",
    ):
        """
        Initialize a SentenceTextSplitter.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            encoding_name: Name of the tiktoken encoding to use
            separator: Sentence separator character
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self.separator = separator
        
        # Load the tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
            
        logger.info(f"Initialized SentenceTextSplitter with chunk_size={chunk_size}, "
                   f"chunk_overlap={chunk_overlap}, encoding={encoding_name}")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on sentences.
        
        Args:
            text: The text to split
            
        Returns:
            List[str]: List of text chunks
        """
        if not text:
            return []
            
        # Split text into sentences
        sentences = [s.strip() + self.separator for s in text.split(self.separator) if s.strip()]
        
        # Initialize variables
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            # Get token count for this sentence
            sentence_tokens = self.tokenizer.encode(sentence)
            sentence_token_count = len(sentence_tokens)
            
            # Check if adding this sentence would exceed chunk size
            if current_token_count + sentence_token_count > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append("".join(current_chunk))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Calculate how many sentences to keep for overlap
                    overlap_token_count = 0
                    overlap_sentences = []
                    
                    for s in reversed(current_chunk):
                        s_tokens = self.tokenizer.encode(s)
                        s_token_count = len(s_tokens)
                        
                        if overlap_token_count + s_token_count <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_token_count += s_token_count
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_token_count = overlap_token_count
                else:
                    # No overlap
                    current_chunk = []
                    current_token_count = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append("".join(current_chunk))
            
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks


def get_text_splitter(
    splitter_type: str = "token",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> TextSplitter:
    """
    Get a text splitter instance.
    
    Args:
        splitter_type: Type of splitter to use ('token' or 'sentence')
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        
    Returns:
        TextSplitter: A text splitter instance
        
    Raises:
        ValueError: If the splitter type is not supported
    """
    if splitter_type == "token":
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == "sentence":
        return SentenceTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unsupported splitter type: {splitter_type}") 