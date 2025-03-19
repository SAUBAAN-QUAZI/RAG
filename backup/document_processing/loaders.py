"""
Document Loaders
-------------
This module contains various document loaders for different file types.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import pypdf
from pypdf import PdfReader

from rag.document_processing.document import Document
from rag.utils import logger, get_file_hash


class PDFLoader:
    """
    Loads text and metadata from PDF documents.
    
    This loader extracts text content and metadata from PDF files
    using the pypdf library.
    """
    
    def __init__(self, extract_images: bool = False, page_batch_size: int = 50):
        """
        Initialize a PDFLoader.
        
        Args:
            extract_images: Whether to extract images from the PDF (not yet implemented)
            page_batch_size: Number of pages to process at once for large PDFs
        """
        self.extract_images = extract_images
        self.page_batch_size = page_batch_size
        
        if extract_images:
            logger.warning("Image extraction is not yet implemented")
        
    def load(self, file_path: Union[str, Path], **metadata) -> Document:
        """
        Load a PDF document and extract its text and metadata.
        
        Args:
            file_path: Path to the PDF file
            **metadata: Additional metadata to include
            
        Returns:
            Document: A Document instance containing the PDF content
            
        Raises:
            ValueError: If the file is not a PDF
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"File {file_path} is not a PDF")
            
        start_time = datetime.now()
        logger.info(f"Loading PDF: {file_path}")
        
        # Extract text content
        text = self._extract_text(file_path)
        logger.info(f"Extracted {len(text):,} characters of text from PDF")
        
        # Extract metadata
        pdf_metadata = self._extract_metadata(file_path)
        logger.info(f"Extracted metadata from PDF: {len(pdf_metadata)} fields")
        
        # Add file metadata
        file_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": "pdf",
            "file_size": file_path.stat().st_size,
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "file_hash": get_file_hash(file_path),
            "creation_date": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            "modification_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "processing_date": datetime.now().isoformat(),
        }
        
        # Merge all metadata
        combined_metadata = {**file_metadata, **pdf_metadata, **metadata}
        
        # Create document
        doc = Document(content=text, metadata=combined_metadata)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"PDF loaded in {processing_time:.2f} seconds, document ID: {doc.doc_id}")
        
        return doc
        
    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text content from a PDF file.
        
        For large PDFs, processes pages in batches to manage memory usage.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            with open(file_path, "rb") as file:
                pdf = PdfReader(file)
                num_pages = len(pdf.pages)
                
                # Initialize variables
                all_text = []
                
                # Log PDF info
                logger.info(f"PDF has {num_pages} pages")
                
                # For large PDFs, process in batches
                if num_pages > self.page_batch_size:
                    logger.info(f"Large PDF detected. Processing in batches of {self.page_batch_size} pages")
                    
                    for batch_start in range(0, num_pages, self.page_batch_size):
                        batch_end = min(batch_start + self.page_batch_size, num_pages)
                        logger.info(f"Processing pages {batch_start+1}-{batch_end} of {num_pages}")
                        
                        batch_text = []
                        for i in range(batch_start, batch_end):
                            page = pdf.pages[i]
                            try:
                                page_text = page.extract_text()
                                if page_text:
                                    batch_text.append(page_text)
                                else:
                                    logger.warning(f"No text extracted from page {i+1}")
                            except Exception as e:
                                logger.warning(f"Error extracting text from page {i+1}: {str(e)}")
                        
                        # Add batch text to overall text
                        all_text.extend(batch_text)
                        
                        # Log batch processing
                        logger.info(f"Processed batch: pages {batch_start+1}-{batch_end}, extracted text: {sum(len(t) for t in batch_text):,} characters")
                else:
                    # Process all pages at once for smaller PDFs
                    for i, page in enumerate(pdf.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                all_text.append(page_text)
                            else:
                                logger.warning(f"No text extracted from page {i+1}")
                        except Exception as e:
                            logger.warning(f"Error extracting text from page {i+1}: {str(e)}")
                
                # Create a single text string from all extracted text
                text = "\n\n".join(all_text)
                
                # Check if we extracted any meaningful text
                if not text.strip():
                    logger.warning("No text extracted from PDF. This may be a scanned document or contains only images.")
                    text = "[PDF contains no extractable text. This may be a scanned document or contains only images.]"
                
                return text
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
            
    def _extract_metadata(self, file_path: Path) -> Dict:
        """
        Extract metadata from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict: Extracted metadata
        """
        try:
            with open(file_path, "rb") as file:
                pdf = PdfReader(file)
                
                # Initialize metadata
                metadata = {}
                
                # Extract document info
                if pdf.metadata:
                    for key, value in pdf.metadata.items():
                        # Remove the leading slash from key if present
                        clean_key = key[1:] if key.startswith("/") else key
                        
                        # Convert value to string if it's not already
                        try:
                            if isinstance(value, (bytes, bytearray)):
                                value = value.decode("utf-8", errors="replace")
                            elif not isinstance(value, str):
                                value = str(value)
                        except Exception:
                            value = str(value)
                            
                        metadata[clean_key] = value
                
                # Add document structure info
                metadata["page_count"] = len(pdf.pages)
                
                # Add page sizes for first few pages
                page_sizes = []
                max_pages_to_check = min(5, len(pdf.pages))
                
                for i in range(max_pages_to_check):
                    page = pdf.pages[i]
                    if page.mediabox:
                        width = round(float(page.mediabox[2]), 2)
                        height = round(float(page.mediabox[3]), 2)
                        page_sizes.append(f"{width}x{height}")
                
                if page_sizes:
                    metadata["page_size_sample"] = page_sizes
                
                return metadata
                
        except Exception as e:
            logger.warning(f"Error extracting metadata from PDF: {str(e)}")
            return {}


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