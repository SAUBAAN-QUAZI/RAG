"""
Document Processing Module
-------------------------
This module contains components for loading, processing, and chunking documents
for use in the RAG system.

Key components:
- Document Loaders: Extract text from various file formats
- Text Splitters: Divide documents into manageable chunks
- Text Cleaners: Remove irrelevant information from text
"""

from rag.document_processing.document import Document, DocumentChunk
from rag.document_processing.loaders import PDFLoader, load_document
from rag.document_processing.splitters import (
    TextSplitter,
    TokenTextSplitter,
    SentenceTextSplitter,
    get_text_splitter
)
from rag.document_processing.processor import DocumentProcessor, process_document

__all__ = [
    'Document',
    'DocumentChunk',
    'PDFLoader',
    'load_document',
    'TextSplitter',
    'TokenTextSplitter',
    'SentenceTextSplitter',
    'get_text_splitter',
    'DocumentProcessor',
    'process_document'
] 