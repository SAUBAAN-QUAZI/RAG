"""
Document Processor
----------------
This module combines document loading and splitting into a single process.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from rag.config import CHUNKS_DIR, DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from rag.document_processing.document import Document, DocumentChunk
from rag.document_processing.loaders import load_document
from rag.document_processing.splitters import get_text_splitter
from rag.utils import logger, save_json


class DocumentProcessor:
    """
    Process documents by loading and splitting them.
    """
    
    def __init__(
        self,
        documents_dir: Union[str, Path] = DOCUMENTS_DIR,
        chunks_dir: Union[str, Path] = CHUNKS_DIR,
        splitter_type: str = "token",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Initialize a DocumentProcessor.
        
        Args:
            documents_dir: Directory to store processed documents
            chunks_dir: Directory to store document chunks
            splitter_type: Type of text splitter to use
            chunk_size: Maximum size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.documents_dir = Path(documents_dir)
        self.chunks_dir = Path(chunks_dir)
        
        # Create directories if they don't exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # Use configuration defaults if not specified
        if chunk_size is None:
            chunk_size = CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = CHUNK_OVERLAP
            
        # Initialize text splitter
        self.text_splitter = get_text_splitter(
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        logger.info(f"Initialized DocumentProcessor with splitter_type={splitter_type}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def process_file(
        self,
        file_path: Union[str, Path],
        save_results: bool = True,
        **metadata
    ) -> Dict[str, Union[Document, List[DocumentChunk]]]:
        """
        Process a document file by loading and splitting it.
        
        Args:
            file_path: Path to the document file
            save_results: Whether to save the processed document and chunks
            **metadata: Additional metadata to include
            
        Returns:
            Dict containing the processed document and chunks
        """
        file_path = Path(file_path)
        logger.info(f"Processing document: {file_path}")
        
        # Load document
        document = load_document(file_path, **metadata)
        logger.info(f"Loaded document with ID: {document.doc_id}")
        
        # Split document into chunks
        chunks = self.text_splitter.split_document(document)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Save results if requested
        if save_results:
            self._save_document(document)
            self._save_chunks(chunks)
        
        return {
            "document": document,
            "chunks": chunks,
        }
    
    def process_text(
        self,
        text: str,
        metadata: Dict = None,
        save_results: bool = True,
    ) -> Dict[str, Union[Document, List[DocumentChunk]]]:
        """
        Process text by creating a document and splitting it.
        
        Args:
            text: Text content to process
            metadata: Metadata for the document
            save_results: Whether to save the processed document and chunks
            
        Returns:
            Dict containing the processed document and chunks
        """
        if metadata is None:
            metadata = {}
            
        logger.info("Processing text input")
        
        # Create document
        document = Document(content=text, metadata=metadata)
        logger.info(f"Created document with ID: {document.doc_id}")
        
        # Split document into chunks
        chunks = self.text_splitter.split_document(document)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Save results if requested
        if save_results:
            self._save_document(document)
            self._save_chunks(chunks)
        
        return {
            "document": document,
            "chunks": chunks,
        }
    
    def _save_document(self, document: Document) -> None:
        """
        Save a document to the documents directory.
        
        Args:
            document: Document to save
        """
        # Create filename based on document ID
        file_path = self.documents_dir / f"{document.doc_id}.json"
        
        # Save document as JSON
        save_json(document.to_dict(), file_path)
        logger.info(f"Document saved to {file_path}")
    
    def _save_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Save document chunks to the chunks directory.
        
        Args:
            chunks: List of chunks to save
        """
        if not chunks:
            return
            
        # Get document ID from the first chunk
        doc_id = chunks[0].metadata.get("doc_id", "unknown")
        
        # Create filename based on document ID
        file_path = self.chunks_dir / f"{doc_id}_chunks.json"
        
        # Save chunks as JSON
        chunks_data = [chunk.to_dict() for chunk in chunks]
        save_json(chunks_data, file_path)
        logger.info(f"Chunks saved to {file_path}")


# Convenience function for processing a document
def process_document(
    file_path: Union[str, Path],
    splitter_type: str = "token",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    save_results: bool = True,
    **metadata
) -> Dict[str, Union[Document, List[DocumentChunk]]]:
    """
    Process a document file by loading and splitting it.
    
    Args:
        file_path: Path to the document file
        splitter_type: Type of text splitter to use
        chunk_size: Maximum size of each chunk in tokens
        chunk_overlap: Number of tokens to overlap between chunks
        save_results: Whether to save the processed document and chunks
        **metadata: Additional metadata to include
        
    Returns:
        Dict containing the processed document and chunks
    """
    # Use configuration defaults if not specified
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP
        
    processor = DocumentProcessor(
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    return processor.process_file(
        file_path=file_path,
        save_results=save_results,
        **metadata,
    ) 