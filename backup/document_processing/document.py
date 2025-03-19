"""
Document Class
-----------
Represents a document and its metadata in the RAG system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from rag.utils import get_file_hash, get_text_hash


@dataclass
class Document:
    """
    Represents a document in the RAG system.
    
    Attributes:
        content: The text content of the document
        metadata: Additional information about the document
        doc_id: Unique identifier for the document
    """
    content: str
    metadata: Dict = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        # Generate a document ID if not provided
        if self.doc_id is None:
            self.doc_id = get_text_hash(self.content)[:16]
            
        # Add timestamp if not present
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()
            
    @classmethod
    def from_file(cls, file_path: Union[str, Path], **metadata) -> "Document":
        """
        Create a Document from a file.
        
        Note: This method is for text files only.
        For binary files like PDFs, use the appropriate loader instead.
        
        Args:
            file_path: Path to the file
            **metadata: Additional metadata to include
            
        Returns:
            Document: A new Document instance
        """
        file_path = Path(file_path)
        
        # Check file type
        if file_path.suffix.lower() == ".pdf":
            raise ValueError(
                "PDF files cannot be read directly. "
                "Use PDFLoader from rag.document_processing.loaders instead."
            )
        
        # Read file content as text
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except UnicodeDecodeError:
            raise ValueError(
                f"Cannot read {file_path} as text. It may be a binary file. "
                "Use the appropriate loader instead."
            )
            
        # Gather basic metadata
        file_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": file_path.suffix.lstrip(".").lower(),
            "file_hash": get_file_hash(file_path),
            "file_size": file_path.stat().st_size,
            "creation_date": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            "modification_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }
        
        # Merge provided metadata with file metadata
        file_metadata.update(metadata)
        
        return cls(content=content, metadata=file_metadata)
    
    def to_dict(self) -> Dict:
        """
        Convert the document to a dictionary.
        
        Returns:
            Dict: Document as a dictionary
        """
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Document":
        """
        Create a Document from a dictionary.
        
        Args:
            data: Dictionary representation of a document
            
        Returns:
            Document: A new Document instance
        """
        return cls(
            content=data["content"],
            metadata=data["metadata"],
            doc_id=data["doc_id"],
        )


@dataclass
class DocumentChunk:
    """
    Represents a chunk of a document in the RAG system.
    
    Attributes:
        content: The text content of the chunk
        metadata: Additional information about the chunk and its source document
        chunk_id: Unique identifier for the chunk
    """
    content: str
    metadata: Dict = field(default_factory=dict)
    chunk_id: Optional[str] = None
    
    def __post_init__(self):
        # Generate a chunk ID if not provided
        if self.chunk_id is None:
            self.chunk_id = get_text_hash(self.content)[:16]
            
        # Add timestamp if not present
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()
            
    def to_dict(self) -> Dict:
        """
        Convert the document chunk to a dictionary.
        
        Returns:
            Dict: DocumentChunk as a dictionary
        """
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DocumentChunk":
        """
        Create a DocumentChunk from a dictionary.
        
        Args:
            data: Dictionary representation of a document chunk
            
        Returns:
            DocumentChunk: A new DocumentChunk instance
        """
        return cls(
            content=data["content"],
            metadata=data["metadata"],
            chunk_id=data["chunk_id"],
        ) 