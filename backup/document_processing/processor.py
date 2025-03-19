"""
Document Processor
----------------
This module combines document loading and splitting into a single process.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

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
        chunk_batch_size: int = 100,  # Process chunks in batches for large documents
        **metadata
    ) -> Dict[str, Union[Document, List[DocumentChunk]]]:
        """
        Process a document file by loading and splitting it into chunks.
        
        Args:
            file_path: Path to the document file
            save_results: Whether to save the processed document and chunks
            chunk_batch_size: Number of chunks to process at once for large documents
            **metadata: Additional metadata to include
            
        Returns:
            Dict containing the processed document and chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Log start of processing
        start_time = datetime.now()
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Load document
            document = load_document(file_path, **metadata)
            document_size = len(document.content)
            logger.info(f"Loaded document with {document_size:,} characters")
            
            # Estimate number of tokens (rough estimate: ~4 chars per token)
            est_tokens = document_size // 4
            logger.info(f"Estimated document size: ~{est_tokens:,} tokens")
            
            # Split document into chunks
            chunks = self.text_splitter.split_document(document)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Save document if requested
            if save_results:
                self._save_document(document)
                
                # For very large documents, save chunks in batches to manage memory
                if len(chunks) > chunk_batch_size:
                    logger.info(f"Document is large. Saving {len(chunks)} chunks in batches of {chunk_batch_size}")
                    for i in range(0, len(chunks), chunk_batch_size):
                        batch = chunks[i:i + chunk_batch_size]
                        # Save this batch only - use a special batch path
                        batch_path = self.chunks_dir / f"{document.doc_id}_chunks_batch_{i//chunk_batch_size}.json"
                        batch_data = [chunk.to_dict() for chunk in batch]
                        save_json(batch_data, batch_path)
                        logger.info(f"Saved batch {i//chunk_batch_size + 1} with {len(batch)} chunks")
                    
                    # Save an index file to track all batches
                    index_path = self.chunks_dir / f"{document.doc_id}_chunks_index.json"
                    batch_count = (len(chunks) + chunk_batch_size - 1) // chunk_batch_size
                    index_data = {
                        "doc_id": document.doc_id,
                        "total_chunks": len(chunks),
                        "batch_count": batch_count,
                        "batch_size": chunk_batch_size,
                        "batches": [f"{document.doc_id}_chunks_batch_{i}.json" for i in range(batch_count)]
                    }
                    save_json(index_data, index_path)
                    logger.info(f"Saved chunk index with {batch_count} batches")
                else:
                    # Save all chunks at once for smaller documents
                    self._save_chunks(chunks)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Document processed in {processing_time:.2f} seconds")
            
            return {
                "document": document,
                "chunks": chunks,
                "processing_time": processing_time,
                "chunk_count": len(chunks)
            }
            
        except Exception as e:
            logger.exception(f"Error processing file {file_path}: {str(e)}")
            raise
    
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