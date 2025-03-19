"""
Text Splitters
-----------
This module contains text splitters for dividing documents into chunks.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import nltk
import tiktoken

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP
from rag.document_processing.document import Document, DocumentChunk
from rag.utils import logger

# Initialize NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer")
    nltk.download('punkt', quiet=True)


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
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        encoding_name: str = "cl100k_base",  # Default for GPT-4, GPT-3.5-Turbo
    ):
        """
        Initialize a TokenTextSplitter.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            encoding_name: Name of the tiktoken encoding to use
        """
        # Use defaults from config if not specified
        self.chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
        self.encoding_name = encoding_name
        
        # Ensure chunk_size and chunk_overlap are valid
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap must be less than chunk_size, got {self.chunk_overlap} >= {self.chunk_size}")
        
        # Load the tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
            
        logger.info(f"Initialized TokenTextSplitter with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}, encoding={encoding_name}")
    
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
            
        # Safety check - ensure chunk_size and chunk_overlap are integers
        # This is a fallback in case the __init__ safeguards somehow failed
        chunk_size = self.chunk_size if isinstance(self.chunk_size, int) else CHUNK_SIZE
        chunk_overlap = self.chunk_overlap if isinstance(self.chunk_overlap, int) else CHUNK_OVERLAP
        
        # Extra logging to debug
        logger.info(f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap} for splitting")
            
        # Encode the text
        tokens = self.tokenizer.encode(text)
        
        # Split tokens into chunks
        chunks = []
        i = 0
        while i < len(tokens):
            # Get chunk tokens
            chunk_end = min(i + chunk_size, len(tokens))
            chunk_tokens = tokens[i:chunk_end]
            
            # Decode chunk
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move to next chunk, considering overlap
            i += chunk_size - chunk_overlap
            
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
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
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
        # Use defaults from config if not specified
        self.chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
        self.encoding_name = encoding_name
        self.separator = separator
        
        # Ensure chunk_size and chunk_overlap are valid
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap must be less than chunk_size, got {self.chunk_overlap} >= {self.chunk_size}")
        
        # Load the tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
            
        logger.info(f"Initialized SentenceTextSplitter with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}, encoding={encoding_name}")
    
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
            
        # Safety check - ensure chunk_size and chunk_overlap are integers
        # This is a fallback in case the __init__ safeguards somehow failed
        chunk_size = self.chunk_size if isinstance(self.chunk_size, int) else CHUNK_SIZE
        chunk_overlap = self.chunk_overlap if isinstance(self.chunk_overlap, int) else CHUNK_OVERLAP
        
        # Extra logging to debug
        logger.info(f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap} for splitting")
            
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
            if current_token_count + sentence_token_count > chunk_size and current_chunk:
                # Save current chunk
                chunks.append("".join(current_chunk))
                
                # Start new chunk with overlap
                if chunk_overlap > 0:
                    # Calculate how many sentences to keep for overlap
                    overlap_token_count = 0
                    overlap_sentences = []
                    
                    for s in reversed(current_chunk):
                        s_tokens = self.tokenizer.encode(s)
                        s_token_count = len(s_tokens)
                        
                        if overlap_token_count + s_token_count <= chunk_overlap:
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


class SemanticTextSplitter(TextSplitter):
    """
    Split text into chunks based on semantic boundaries.
    
    This splitter tries to keep semantically related content together by
    analyzing paragraph breaks, section headings, and content transitions.
    It combines semantic understanding with token-based limits for better chunks.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        encoding_name: str = "cl100k_base",
        heading_pattern: str = r'^(#+\s+.*?|.*?\n[-=]+\n)',  # Markdown/RST-style headings
        paragraph_break: str = "\n\n",
    ):
        """
        Initialize a SemanticTextSplitter.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            encoding_name: Name of the tiktoken encoding to use
            heading_pattern: Regex pattern to identify section headings
            paragraph_break: String that denotes paragraph breaks
        """
        # Use defaults from config if not specified
        self.chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
        self.encoding_name = encoding_name
        self.heading_pattern = re.compile(heading_pattern, re.MULTILINE)
        self.paragraph_break = paragraph_break
        
        # Ensure chunk_size and chunk_overlap are valid
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap must be less than chunk_size, got {self.chunk_overlap} >= {self.chunk_size}")
        
        # Load tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
            
        logger.info(f"Initialized SemanticTextSplitter with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}, encoding={encoding_name}")
    
    def _find_section_boundaries(self, text: str) -> List[int]:
        """
        Find section boundaries based on headings and paragraph breaks.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of section boundary indices
        """
        # Find all headings
        heading_matches = list(self.heading_pattern.finditer(text))
        heading_positions = [match.start() for match in heading_matches]
        
        # Find paragraph breaks
        paragraph_positions = [m.start() for m in re.finditer(re.escape(self.paragraph_break), text)]
        
        # Combine and sort all boundary positions
        boundaries = sorted(set([0] + heading_positions + paragraph_positions + [len(text)]))
        return boundaries
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on semantic boundaries.
        
        Args:
            text: The text to split
            
        Returns:
            List[str]: List of text chunks
        """
        if not text:
            return []
        
        # Find section boundaries
        boundaries = self._find_section_boundaries(text)
        
        # Split text into semantic sections
        sections = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            section = text[start:end].strip()
            if section:  # Skip empty sections
                sections.append(section)
        
        # If no sections were found, fall back to simple paragraph splitting
        if not sections:
            sections = [p for p in text.split(self.paragraph_break) if p.strip()]
            if not sections:
                # Last resort: just return the whole text
                return [text]
        
        # Process sections to respect token limits
        chunks = []
        current_chunk = []
        current_chunk_tokens = 0
        
        for section in sections:
            # Tokenize the section
            section_tokens = self.tokenizer.encode(section)
            section_token_count = len(section_tokens)
            
            # If section is too big on its own, recursively split it
            if section_token_count > self.chunk_size:
                # Use sentence tokenizer to break down large sections
                sentences = nltk.sent_tokenize(section)
                section_chunks = []
                current_section_chunk = []
                current_section_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens = self.tokenizer.encode(sentence)
                    sentence_token_count = len(sentence_tokens)
                    
                    # If adding this sentence would exceed chunk size, start a new chunk
                    if current_section_tokens + sentence_token_count > self.chunk_size and current_section_chunk:
                        section_chunks.append(" ".join(current_section_chunk))
                        current_section_chunk = [sentence]
                        current_section_tokens = sentence_token_count
                    else:
                        current_section_chunk.append(sentence)
                        current_section_tokens += sentence_token_count
                
                # Add the last section chunk if it has content
                if current_section_chunk:
                    section_chunks.append(" ".join(current_section_chunk))
                
                # Add these chunks to our result
                chunks.extend(section_chunks)
            
            # If section fits in current chunk, add it
            elif current_chunk_tokens + section_token_count <= self.chunk_size:
                current_chunk.append(section)
                current_chunk_tokens += section_token_count
            
            # Otherwise start a new chunk
            else:
                if current_chunk:
                    chunks.append(self.paragraph_break.join(current_chunk))
                current_chunk = [section]
                current_chunk_tokens = section_token_count
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(self.paragraph_break.join(current_chunk))
        
        # Create overlap between chunks if specified
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            
            for i, chunk in enumerate(chunks):
                if i == 0:
                    # First chunk remains unchanged
                    overlapped_chunks.append(chunk)
                    continue
                
                # Get the end of the previous chunk for overlap
                prev_chunk = chunks[i-1]
                prev_tokens = self.tokenizer.encode(prev_chunk)
                
                if len(prev_tokens) <= self.chunk_overlap:
                    # If previous chunk is smaller than overlap, just include all of it
                    overlapped_chunks.append(f"{prev_chunk}{self.paragraph_break}{chunk}")
                else:
                    # Take the last chunk_overlap tokens from previous chunk
                    overlap_tokens = prev_tokens[-self.chunk_overlap:]
                    overlap_text = self.tokenizer.decode(overlap_tokens)
                    
                    # Add overlap to the beginning of current chunk
                    if not chunk.startswith(overlap_text):
                        overlapped_chunks.append(f"{overlap_text}{self.paragraph_break}{chunk}")
                    else:
                        overlapped_chunks.append(chunk)
            
            chunks = overlapped_chunks
        
        logger.info(f"Split text into {len(chunks)} semantic chunks")
        return chunks


def get_text_splitter(
    splitter_type: str = "token",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> TextSplitter:
    """
    Get a text splitter instance.
    
    Args:
        splitter_type: Type of splitter to use ('token', 'sentence', or 'semantic')
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        
    Returns:
        TextSplitter: A text splitter instance
        
    Raises:
        ValueError: If the splitter type is not supported
    """
    # Use defaults from config if not specified
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP

    if splitter_type == "token":
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == "sentence":
        return SentenceTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == "semantic":
        return SemanticTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unsupported splitter type: {splitter_type}") 