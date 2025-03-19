"""
Document Processing Module
-------------------------
Functions for processing documents into chunks for retrieval.

This module handles:
1. Loading documents from various file types
2. Splitting documents into manageable chunks
3. Extracting metadata from documents
4. Detecting document structure (chapters, sections, etc.)
5. Generating unique IDs for documents and chunks
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from rag.utils import logger
from rag.document.text_splitter import TextSplitter, TokenTextSplitter, SentenceTextSplitter

class DocumentProcessor:
    """
    Process documents for retrieval.
    
    This class handles:
    1. Loading documents from various file types
    2. Splitting documents into manageable chunks
    3. Extracting metadata from documents
    4. Generating unique IDs for documents and chunks
    """
    
    def __init__(self, 
                 splitter_type: str = "token", 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            splitter_type: Type of text splitter ('token' or 'sentence')
            chunk_size: Size of chunks (in tokens or sentences)
            chunk_overlap: Overlap between chunks (in tokens or sentences)
        """
        self.splitter_type = splitter_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        if splitter_type == "token":
            self.splitter = TokenTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            logger.info(f"Initialized TokenTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, encoding=cl100k_base")
        elif splitter_type == "sentence":
            self.splitter = SentenceTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            logger.info(f"Initialized SentenceTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")
        
        logger.info(f"Initialized DocumentProcessor with splitter_type={splitter_type}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def _extract_document_structure(self, text: str) -> Dict[str, Any]:
        """
        Extract document structure information like chapters and sections.
        
        This helps with organizing content and improving retrievability of 
        specific document regions when users ask for them.
        
        Args:
            text: The document text
            
        Returns:
            Dictionary with structure information
        """
        structure_info = {
            "has_chapters": False,
            "chapter_count": 0,
            "chapters": [],
            "sections": []
        }
        
        # Look for chapter patterns
        chapter_patterns = [
            r"(?:Chapter|CHAPTER)\s+(\d+|[IVXivx]+)[\s\n:]+([^\n]+)?",  # Chapter 1: Introduction
            r"(?:^|\n)(\d+|[IVXivx]+)[\.\s]+([A-Z][^\n]+)(?:\n|$)",     # 1. INTRODUCTION
        ]
        
        sections_found = []
        chapters_found = []
        
        # Extract chapters
        for pattern in chapter_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                chapter_num = match.group(1)
                chapter_title = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else ""
                
                # Convert Roman numerals to integers if needed
                try:
                    if all(c in "IVXLCDMivxlcdm" for c in chapter_num):
                        # This is a Roman numeral
                        chapter_num = chapter_num.upper()  # Standardize to uppercase
                    else:
                        # This is a digit
                        chapter_num = int(chapter_num)
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass
                
                chapters_found.append({
                    "number": chapter_num,
                    "title": chapter_title,
                    "position": match.start()
                })
        
        # Sort chapters by position in text
        chapters_found.sort(key=lambda x: x["position"])
        
        # Extract sections if we found chapters
        if chapters_found:
            structure_info["has_chapters"] = True
            structure_info["chapter_count"] = len(chapters_found)
            structure_info["chapters"] = chapters_found
            
            # Look for section patterns
            section_patterns = [
                r"(?:^|\n)(\d+\.\d+)[\.\s]+([^\n]+)",  # 1.2 Section Title
                r"(?:Section|SECTION)\s+(\d+\.\d+)[\s\n:]+([^\n]+)?",  # Section 1.2: Title
            ]
            
            for pattern in section_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    section_num = match.group(1)
                    section_title = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else ""
                    
                    # Try to identify which chapter this section belongs to
                    chapter_idx = None
                    for i in range(len(chapters_found) - 1):
                        if (chapters_found[i]["position"] < match.start() and 
                            chapters_found[i+1]["position"] > match.start()):
                            chapter_idx = i
                            break
                    
                    # If it's after the last chapter
                    if chapter_idx is None and chapters_found and match.start() > chapters_found[-1]["position"]:
                        chapter_idx = len(chapters_found) - 1
                    
                    sections_found.append({
                        "number": section_num,
                        "title": section_title,
                        "position": match.start(),
                        "chapter_idx": chapter_idx
                    })
            
            # Sort sections by position
            sections_found.sort(key=lambda x: x["position"])
            structure_info["sections"] = sections_found
        
        return structure_info

    def _enrich_chunk_metadata(self, chunk: Dict[str, Any], doc_structure: Dict[str, Any], 
                             text_position: int, text_end_position: int) -> Dict[str, Any]:
        """
        Enrich chunk metadata with structural information.
        
        Adds chapter and section information to chunk metadata based on
        the document structure and the chunk's position in the text.
        
        Args:
            chunk: The chunk to enrich
            doc_structure: Document structure information
            text_position: Start position of chunk in the original text
            text_end_position: End position of chunk in the original text
            
        Returns:
            Chunk with enriched metadata
        """
        # Start with existing metadata
        metadata = chunk.get("metadata", {}).copy()
        
        # Add structural information if available
        if doc_structure["has_chapters"]:
            # Find which chapter this chunk belongs to
            chapter_idx = None
            for i, chapter in enumerate(doc_structure["chapters"]):
                # If this is the last chapter
                if i == len(doc_structure["chapters"]) - 1:
                    if chapter["position"] <= text_position:
                        chapter_idx = i
                        break
                # If between this chapter and the next
                elif (chapter["position"] <= text_position and 
                      doc_structure["chapters"][i+1]["position"] > text_position):
                    chapter_idx = i
                    break
            
            # If we found a chapter for this chunk
            if chapter_idx is not None:
                chapter = doc_structure["chapters"][chapter_idx]
                metadata["chapter"] = chapter["number"]
                if chapter["title"]:
                    metadata["chapter_title"] = chapter["title"]
                
                # Find sections within this chapter
                if doc_structure["sections"]:
                    relevant_sections = [
                        s for s in doc_structure["sections"] 
                        if s["chapter_idx"] == chapter_idx and s["position"] <= text_position
                    ]
                    
                    if relevant_sections:
                        # Use the most specific (last) section that contains this chunk
                        section = relevant_sections[-1]
                        metadata["section"] = section["number"]
                        if section["title"]:
                            metadata["section_title"] = section["title"]
        
        # Update the chunk with enriched metadata
        chunk["metadata"] = metadata
        return chunk

    def _process_document(self, document_path: str, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document text into chunks with metadata.
        
        Args:
            document_path: Path to the document
            text: The raw document text
            metadata: Additional metadata for the document
            
        Returns:
            Dictionary containing document ID, metadata, and chunks
        """
        try:
            logger.info(f"Loaded document with {len(text)} characters")
            # Estimate token count for logging
            estimated_tokens = len(text) // 4
            logger.info(f"Estimated document size: ~{estimated_tokens:,} tokens")
            
            # Extract document structure information
            doc_structure = self._extract_document_structure(text)
            if doc_structure["has_chapters"]:
                logger.info(f"Detected {doc_structure['chapter_count']} chapters in document")
                # Add to document metadata
                metadata["chapter_count"] = doc_structure["chapter_count"]
                metadata["has_chapters"] = True
                if doc_structure["sections"]:
                    metadata["has_sections"] = True
                    metadata["section_count"] = len(doc_structure["sections"])
            
            # Log splitting configuration
            logger.info(f"Using chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap} for splitting")
            
            # Split the text
            chunks = self.splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Determine document ID - either use existing one or generate a new one
            if "document_id" in metadata:
                document_id = metadata["document_id"]
            else:
                document_id = self._generate_document_id(text[:5000], metadata)
            
            # Process chunks
            processed_chunks = []
            
            # Track the running position in the text for structural information
            running_position = 0
            prev_chunk_end = 0
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Try to find the position of this chunk in the original text
                # We start searching from the end of the previous chunk for efficiency
                chunk_position = text.find(chunk, prev_chunk_end)
                
                # If not found from previous chunk (can happen with some splitters),
                # search from beginning
                if chunk_position == -1:
                    chunk_position = text.find(chunk)
                    
                # If still not found, use the running position as an approximation
                if chunk_position == -1:
                    chunk_position = running_position
                else:
                    # Update running position based on found position
                    running_position = chunk_position
                    prev_chunk_end = chunk_position + len(chunk)
                
                # Calculate the end position
                chunk_end_position = chunk_position + len(chunk)
                
                # Generate a unique ID for the chunk
                chunk_id = self._generate_chunk_id(chunk)
                
                # Create chunk metadata
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": document_path,
                    **metadata,  # Include all document metadata
                }
                
                # Add page number if available (for PDFs)
                if "page_mapping" in metadata:
                    # Find the closest page from the mapping
                    closest_page = None
                    best_distance = float('inf')
                    
                    for position, page in metadata["page_mapping"].items():
                        position = int(position)  # Convert string key to int
                        distance = abs(position - chunk_position)
                        if distance < best_distance:
                            best_distance = distance
                            closest_page = page
                    
                    if closest_page is not None:
                        chunk_metadata["page_number"] = closest_page
                
                # Create the chunk object
                chunk_obj = {
                    "chunk_id": chunk_id,
                    "content": chunk,
                    "metadata": chunk_metadata,
                }
                
                # Enrich with document structure information
                chunk_obj = self._enrich_chunk_metadata(chunk_obj, doc_structure, 
                                                    chunk_position, chunk_end_position)
                
                processed_chunks.append(chunk_obj)
            
            logger.info(f"Split document into {len(processed_chunks)} chunks")
            
            return {
                "document_id": document_id,
                "metadata": metadata,
                "chunks": processed_chunks,
                "structure": doc_structure
            }
            
        except Exception as e:
            logger.exception(f"Error processing document text: {e}")
            raise 

    def _generate_document_id(self, text_sample: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a unique document ID based on content and metadata.
        
        Args:
            text_sample: Sample of document text
            metadata: Document metadata
            
        Returns:
            Unique document ID
        """
        # Create a string that combines text sample and important metadata
        id_string = text_sample
        
        # Add important metadata that would distinguish this document
        if "title" in metadata:
            id_string += metadata["title"]
        if "author" in metadata:
            id_string += metadata["author"]
        if "file_hash" in metadata:
            # If we have a file hash, prioritize that
            return metadata["file_hash"][:16]
        
        # Generate hash
        hash_obj = hashlib.sha256(id_string.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Use first 16 chars of hash
        
    def _generate_chunk_id(self, chunk_text: str) -> str:
        """
        Generate a unique ID for a text chunk.
        
        Args:
            chunk_text: Text content of the chunk
            
        Returns:
            Unique chunk ID
        """
        # Hash the chunk text to create a unique identifier
        hash_obj = hashlib.sha256(chunk_text.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Use first 16 chars of hash

    def process_file(self, file_path: str, **metadata) -> Dict[str, Any]:
        """
        Process a file into chunks with metadata.
        
        Args:
            file_path: Path to the document file
            **metadata: Additional metadata for the document
            
        Returns:
            Dictionary containing document ID, metadata, and chunks
        """
        from rag.document.loaders import load_document
        
        try:
            # Ensure file path is resolved
            file_path = str(Path(file_path).resolve())
            logger.info(f"Processing file: {file_path}")
            
            # Load the document
            doc_data = load_document(file_path)
            
            # Extract content and combine with provided metadata
            text = doc_data["text"]
            combined_metadata = {**doc_data["metadata"], **metadata}
            
            # Process the document
            result = self._process_document(file_path, text, combined_metadata)
            logger.info(f"Document processed: {result['document_id']}, generated {len(result['chunks'])} chunks")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error processing file {file_path}: {e}")
            raise 