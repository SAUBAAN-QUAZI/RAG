"""
Document Loaders
-------------
This module contains various document loaders for different file types.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pypdf
from pypdf import PdfReader

from rag.document_processing.document import Document
from rag.utils import logger


class PDFLoader:
    """
    Loads text content from PDF files.
    
    This loader extracts text from PDF files using pypdf.
    """
    
    def __init__(self, extract_images: bool = False):
        """
        Initialize a PDF loader.
        
        Args:
            extract_images: Whether to extract and process images in the PDF
                           (Not implemented in this version)
        """
        self.extract_images = extract_images
        if extract_images:
            logger.warning("Image extraction not implemented yet. Ignoring.")
    
    def load(self, file_path: Union[str, Path], **metadata) -> Document:
        """
        Load and process a PDF file.
        
        Args:
            file_path: Path to the PDF file
            **metadata: Additional metadata to include with the document
            
        Returns:
            Document: A document with the extracted text and metadata
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a PDF
        """
        file_path = Path(file_path)
        
        # Verify file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file type
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got {file_path.suffix}")
            
        logger.info(f"Loading PDF: {file_path}")
        
        # Extract text from PDF
        extracted_text = self._extract_text(file_path)
        
        # Get PDF metadata
        pdf_metadata = self._extract_metadata(file_path)
        
        # Combine metadata
        combined_metadata = {**pdf_metadata, **metadata}
        
        # Create document
        document = Document(content=extracted_text, metadata=combined_metadata)
        
        logger.info(f"Successfully loaded PDF: {file_path}")
        return document
    
    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        text = ""
        
        try:
            reader = PdfReader(file_path)
            
            # Get total number of pages
            num_pages = len(reader.pages)
            
            # Extract text from each page
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                
                # Skip empty pages
                if not page_text:
                    logger.warning(f"Page {i+1}/{num_pages} is empty or could not be read")
                    continue
                    
                # Add page number as a heading
                text += f"\n\n--- Page {i+1} ---\n\n"
                text += page_text
                
            if not text.strip():
                logger.warning("No text could be extracted from the PDF")
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
            
        return text
    
    def _extract_metadata(self, file_path: Path) -> Dict:
        """
        Extract metadata from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict: Extracted metadata
        """
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": "pdf",
        }
        
        try:
            reader = PdfReader(file_path)
            
            # Get document info
            info = reader.metadata
            
            if info:
                # Add standard PDF metadata if available
                for key, value in info.items():
                    # Remove the leading slash from keys
                    clean_key = key.strip("/").lower() if isinstance(key, str) else key
                    
                    # Only add non-empty values
                    if value and str(value).strip():
                        metadata[clean_key] = str(value)
            
            # Add document structure info
            metadata["page_count"] = len(reader.pages)
            
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF: {e}")
            
        return metadata


def load_document(file_path: Union[str, Path], **metadata) -> Document:
    """
    Load a document from a file, automatically selecting the appropriate loader.
    
    Args:
        file_path: Path to the document file
        **metadata: Additional metadata to include with the document
        
    Returns:
        Document: A document with the extracted text and metadata
        
    Raises:
        ValueError: If the file type is not supported
    """
    file_path = Path(file_path)
    
    # Get file extension
    file_ext = file_path.suffix.lower()
    
    # Select appropriate loader based on file extension
    if file_ext == ".pdf":
        loader = PDFLoader()
        return loader.load(file_path, **metadata)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}") 