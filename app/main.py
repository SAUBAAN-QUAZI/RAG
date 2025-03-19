"""
FastAPI Application
---------------
This module implements the RESTful API for the RAG system.
"""

import os
import secrets
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import time
import uuid
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.config import (
    API_HOST, API_PORT, DEBUG, ALLOW_CORS, CORS_ORIGINS, 
    DOCUMENTS_DIR, CHUNKS_DIR, VECTORS_DIR,
    OPENAI_API_KEY, DATA_DIR, EMBEDDING_MODEL, MAX_RESPONSE_TOKENS, 
    USE_RAGIE as RAGIE_ENABLED
)
# Only import the Ragie agent, not the standard RAG agent
# from rag.retrieval.rag_agent import RAGAgent
from rag.retrieval.ragie_agent import RagieRAGAgent
from rag.utils import logger
from rag.integrations.ragie import RagieClient

# Import webhook router if available
try:
    from app.webhooks import webhooks_router
    HAS_WEBHOOKS = True
except ImportError:
    HAS_WEBHOOKS = False
    logger.warning("Webhook module not available, webhook endpoints will not be registered")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with documentation
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API for document Q&A",
    version="1.0.0",
    openapi_tags=[
        {"name": "health", "description": "Health check endpoints"},
        {"name": "documents", "description": "Document management endpoints"},
        {"name": "query", "description": "Document query endpoints"},
    ]
)

# Add CORS middleware if enabled
if ALLOW_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Middleware for request timing and logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to track request processing time and log requests
    """
    start_time = time.time()
    
    # Get client IP and requested path for logging
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    method = request.method
    
    # Log the incoming request
    logger.info(f"Request: {method} {path} from {client_ip}")
    
    # Process the request
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log completion with status code and time
        status_code = response.status_code
        logger.info(f"Response: {status_code} for {method} {path} in {process_time:.3f}s")
        
        return response
    except Exception as e:
        # Log error if request processing fails
        process_time = time.time() - start_time
        logger.error(f"Error processing {method} {path}: {str(e)} after {process_time:.3f}s")
        raise

# Register webhook router if available
if "webhooks_router" in locals():
    app.include_router(webhooks_router)
    
# Initialize document processing tracker
document_processing_status = {}

# Initialize RAG agent
rag_agent = None
try:
    if RAGIE_ENABLED:
        logger.info("Initializing Ragie-based RAG agent")
        try:
            rag_agent = RagieRAGAgent()
            logger.info("Ragie RAG agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ragie RAG agent: {e}")
            logger.error("No fallback available as Ragie-only mode is configured")
    else:
        logger.warning("Ragie integration is disabled (USE_RAGIE=False) but system is configured for Ragie-only mode")
        logger.warning("Please set USE_RAGIE=True in your .env file")
except Exception as e:
    logger.error(f"Failed to initialize RAG agent: {e}")
    # Continue without agent - endpoints will handle the None case

# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes

# Function to process document in the background
def process_document_task(file_path: str, doc_id: str, metadata: dict):
    """
    Process a document in the background
    """
    try:
        logger.info(f"Starting background processing of document {doc_id}")
        document_processing_status[doc_id] = "processing"
        
        # Process the document using the appropriate agent
        if RAGIE_ENABLED and isinstance(rag_agent, RagieRAGAgent):
            # For Ragie-based agent, add document with metadata
            result = rag_agent.add_document(
                file_path=file_path,
                document_id=doc_id,
                metadata=metadata
            )
            # Store the Ragie document ID in our mapping
            metadata["ragie_document_id"] = result["id"]
        else:
            # For standard agent, process normally
            result = rag_agent.add_document(file_path, **metadata)
        
        # Update the status
        document_processing_status[doc_id] = "complete"
        logger.info(f"Document processing completed for {doc_id}: {result}")
    except Exception as e:
        # Update status on error
        document_processing_status[doc_id] = "error"
        
        # Clean up the temporary file in case of processing error
        if os.path.exists(file_path):
            os.unlink(file_path)
            
        logger.exception(f"Error processing document: {e}")

# Function to process a single document from a batch
def process_batch_document_task(file_path: str, doc_id: str, metadata: dict, file_result: dict):
    """
    Process a single document from a batch in the background
    """
    try:
        logger.info(f"Starting background processing of batch document {doc_id}")
        document_processing_status[doc_id] = "processing"
        
        # Process the document
        result = rag_agent.add_document(file_path, **metadata)
        
        # Update the status
        document_processing_status[doc_id] = "complete"
        
        # Update file result with success info
        file_result["status"] = "success"
        file_result["details"].update({
            "message": f"Document processed successfully.",
            "chunk_count": result.get("chunk_count", 0),
        })
        
        logger.info(f"Batch document processing completed for {doc_id}: {result}")
    except Exception as e:
        # Update status on error
        document_processing_status[doc_id] = "error"
        
        # Update file result with error info
        file_result["status"] = "error"
        file_result["details"] = {
            "error": f"Error processing document: {str(e)}"
        }
        
        # Clean up the temporary file in case of processing error
        if os.path.exists(file_path):
            os.unlink(file_path)
            
        logger.exception(f"Error processing batch document: {e}")


class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = Field(default=None, description="Optional list of document IDs to search within")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter for document selection")
    rerank: Optional[bool] = Field(default=True, description="Whether to rerank results (Ragie-specific)")
    top_k: Optional[int] = Field(default=None, description="Number of chunks to retrieve")
    show_timings: bool = Field(default=False, description="Whether to include timing information in the response")


class QueryResponse(BaseModel):
    """
    Response model for a query.
    """
    answer: str = Field(..., description="Generated answer")


class MonitoringResponse(BaseModel):
    """
    Response model for system monitoring.
    """
    status: str = Field(..., description="Overall system status")
    components: Dict = Field(..., description="Status of individual components")
    metrics: Dict = Field(..., description="Performance metrics")


@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the RAG API", "version": "0.1.0"}


@app.post("/api/query", response_model=Dict)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question and get an answer based on document context.
    
    Args:
        request: The query request with the question and optional parameters
        
    Returns:
        Dict containing the response with answer and optional context
        
    Raises:
        HTTPException: If the query processing fails
    """
    try:
        # Ensure Ragie integration is enabled
        if not RAGIE_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="RAG system is configured for Ragie-only mode but Ragie is disabled. Please enable Ragie in configuration."
            )
            
        try:
            # Create a Ragie RAG agent for handling this query
            ragie_agent = RagieRAGAgent()
            
            # Process the query using Ragie
            result = ragie_agent.query(
                query=request.query,
                document_ids=request.document_ids,
                filter_metadata=request.metadata_filter,
                top_k=request.top_k,
                show_timings=request.show_timings
            )
            
            return result
        except Exception as e:
            logger.exception(f"Error processing query with Ragie: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing query with Ragie: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/documents")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """
    Upload a document for processing and indexing.
    
    Args:
        background_tasks: FastAPI background tasks
        file: The document file to upload
        title: Optional document title
        author: Optional document author
        description: Optional document description
        
    Returns:
        Dict with status and document ID
        
    Raises:
        HTTPException: If the upload fails
    """
    try:
        # Validate the file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension != '.pdf':
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported at this time"
            )
        
        # Create a unique document ID
        doc_id = f"doc_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Prepare metadata with any provided fields
        metadata = {}
        if title:
            metadata["title"] = title
        else:
            # Use filename as default title
            metadata["title"] = Path(file.filename).stem
            
        if author:
            metadata["author"] = author
        if description:
            metadata["description"] = description
            
        # Add the original filename to metadata
        metadata["filename"] = file.filename
        
        # Check if we're using Ragie
        if RAGIE_ENABLED:
            try:
                # Read the file content
                file_content = await file.read()
                
                # Initialize the Ragie client
                ragie_client = RagieClient()
                
                # Upload the document directly (synchronous in this case)
                result = ragie_client.upload_document_from_bytes(
                    file_content=file_content,
                    file_name=file.filename,
                    metadata=metadata
                )
                
                return {
                    "message": "Document submitted for processing",
                    "document_id": doc_id,
                    "metadata": {
                        "ragie_document_id": result["id"],
                        **metadata
                    },
                    "status": "processing"
                }
            except Exception as e:
                logger.exception(f"Error uploading document to Ragie: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error uploading document to Ragie: {str(e)}"
                )
        else:
            # Use the original document processing pipeline if Ragie is not enabled
            # First save the file
            temp_path = Path(DOCUMENTS_DIR) / f"{doc_id}{file_extension}"
            
            # Create directory if it doesn't exist
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the uploaded file
            file_content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(file_content)
                
            # Process the document in the background
            background_tasks.add_task(
                process_document_task,
                str(temp_path),
                doc_id,
                metadata
            )
            
            return {
                "message": "Document submitted for processing",
                "document_id": doc_id,
                "status": "processing"
            }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )


@app.get("/documents/{document_id}/status")
async def check_document_status(document_id: str):
    """
    Check the processing status of a document.
    
    This endpoint allows clients to poll for the completion status of document processing.
    """
    # Check if RAG agent was successfully initialized
    if rag_agent is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system is not available. Check server logs for initialization errors."
        )
    
    try:
        # First check our in-memory tracker
        if document_id in document_processing_status:
            status = document_processing_status[document_id]
            
            if status == "complete":
                return {
                    "document_id": document_id,
                    "status": "complete",
                    "message": "Document has been fully processed and is ready for querying."
                }
            elif status == "error":
                return {
                    "document_id": document_id,
                    "status": "error",
                    "message": "Document processing failed. Please try uploading again."
                }
            elif status == "processing":
                return {
                    "document_id": document_id,
                    "status": "processing",
                    "message": "Document is still being processed. Please check again later."
                }
            # For "uploading" status or any other status, continue with checks below
        
        # Check if document exists in the vector store (as a backup)
        document_exists = rag_agent.retriever.vector_store.document_exists(document_id)
        
        if document_exists:
            # Document exists in the vector store, which means processing is complete
            # Also update our tracker
            document_processing_status[document_id] = "complete"
            return {
                "document_id": document_id,
                "status": "complete",
                "message": "Document has been fully processed and is ready for querying."
            }
        
        # Check if document file exists (meaning it was uploaded but processing may not be complete)
        # Look for either the temporary file or the processed JSON file
        temp_path = Path(DOCUMENTS_DIR) / f"{document_id}"
        json_path = Path(DOCUMENTS_DIR) / f"{document_id}.json"
        
        # Get creation time of the temp file if it exists
        created_time = None
        if temp_path.exists():
            created_time = temp_path.stat().st_ctime
        elif json_path.exists():
            created_time = json_path.stat().st_ctime
        
        if temp_path.exists() or json_path.exists():
            # Document exists but processing may still be ongoing
            
            # Check if the document has been processing for too long (over 5 minutes)
            # This prevents endless polling loops
            if created_time and (time.time() - created_time) > 300:  # 5 minutes timeout
                # Update our tracker
                document_processing_status[document_id] = "timeout"
                return {
                    "document_id": document_id,
                    "status": "timeout",
                    "message": "Document processing is taking longer than expected. It may still be processing in the background, or there might be an issue with the document."
                }
                
            return {
                "document_id": document_id,
                "status": "processing",
                "message": "Document is still being processed. Please check again later."
            }
        
        # Document not found in tracker or on disk
        # Check if it was previously tracked but now files are gone
        if document_id in document_processing_status:
            status = document_processing_status[document_id]
            if status == "error":
                return {
                    "document_id": document_id,
                    "status": "error",
                    "message": "Document processing failed. Please try uploading again."
                }
            else:
                # Was in the tracker but files are gone - something went wrong
                document_processing_status[document_id] = "error"
                return {
                    "document_id": document_id,
                    "status": "error",
                    "message": "Document files not found. Processing may have failed."
                }
        
        # Document truly not found
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID {document_id} not found."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error checking document status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/batch")
async def upload_multiple_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    title_prefix: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """
    Upload multiple documents for processing and indexing.
    
    Args:
        background_tasks: FastAPI background tasks
        files: List of document files to upload
        title_prefix: Optional prefix for document titles
        author: Optional document author
        description: Optional document description
        
    Returns:
        Dict with status and results for each document
        
    Raises:
        HTTPException: If the upload fails
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files were provided"
        )
    
    # Check if there are PDF files
    pdf_files = [file for file in files if file.filename.lower().endswith('.pdf')]
    
    if not pdf_files:
        raise HTTPException(
            status_code=400,
            detail="No PDF files were provided. Only PDF files are supported at this time."
        )
    
    # Process PDF files
    results = []
    
    try:
        for file in pdf_files:
            # Create a unique document ID
            doc_id = f"doc_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Prepare metadata with any provided fields
            metadata = {}
            
            # Use title prefix + filename if provided, otherwise just filename
            if title_prefix:
                metadata["title"] = f"{title_prefix} - {Path(file.filename).stem}"
            else:
                metadata["title"] = Path(file.filename).stem
                
            if author:
                metadata["author"] = author
            if description:
                metadata["description"] = description
                
            # Add the original filename to metadata
            metadata["filename"] = file.filename
            
            # Track the result for this file
            file_result = {
                "id": doc_id,
                "filename": file.filename,
                "status": "processing"
            }
            
            # Check if we're using Ragie
            if RAGIE_ENABLED:
                try:
                    # Read the file content
                    file_content = await file.read()
                    
                    # Initialize the Ragie client
                    ragie_client = RagieClient()
                    
                    # Upload the document to Ragie
                    result = ragie_client.upload_document_from_bytes(
                        file_content=file_content,
                        file_name=file.filename,
                        metadata=metadata
                    )
                    
                    # Update the file result with Ragie info
                    file_result["ragie_document_id"] = result["id"]
                    file_result["status"] = "processing"
                    file_result["message"] = "Document submitted to Ragie for processing"
                except Exception as e:
                    logger.exception(f"Error uploading {file.filename} to Ragie: {e}")
                    file_result["status"] = "error"
                    file_result["message"] = f"Upload failed: {str(e)}"
            else:
                # Use the original document processing pipeline if Ragie is not enabled
                try:
                    # First save the file
                    file_extension = Path(file.filename).suffix.lower()
                    temp_path = Path(DOCUMENTS_DIR) / f"{doc_id}{file_extension}"
                    
                    # Create directory if it doesn't exist
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Read and save the file
                    file_content = await file.read()
                    with open(temp_path, "wb") as f:
                        f.write(file_content)
                        
                    # Process the document in the background
                    background_tasks.add_task(
                        process_batch_document_task,
                        str(temp_path),
                        doc_id,
                        metadata,
                        file_result
                    )
                except Exception as e:
                    logger.exception(f"Error saving {file.filename}: {e}")
                    file_result["status"] = "error"
                    file_result["message"] = f"Save failed: {str(e)}"
            
            # Add to results
            results.append(file_result)
        
        # Return the batch results
        return {
            "message": f"Batch upload initiated with {len(results)} documents",
            "results": results
        }
    except Exception as e:
        logger.exception(f"Error in batch upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch upload: {str(e)}"
        )


@app.get("/monitoring", response_model=MonitoringResponse)
async def monitor_system():
    """
    Get monitoring information about the RAG system.
    
    Returns detailed status and metrics about all components.
    """
    # Check if RAG agent was successfully initialized
    if rag_agent is None:
        return {
            "status": "critical",
            "components": {
                "rag_agent": {
                    "status": "failed",
                    "error": "RAG agent failed to initialize. Check server logs."
                }
            },
            "metrics": {}
        }
        
    try:
        # Check vector store statistics
        vector_store = rag_agent.retriever.vector_store
        vector_stats = vector_store.get_collection_stats()
        
        # Get vector store metrics (if available)
        vector_metrics = {}
        if hasattr(vector_store, 'get_metrics'):
            vector_metrics = vector_store.get_metrics()
        
        # Check embedding service
        embedding_service = rag_agent.retriever.embedding_service
        
        # System status check
        system_status = "healthy"
        components_status = {
            "vector_store": {
                "status": "healthy", 
                "type": vector_store.__class__.__name__,
                "stats": vector_stats
            },
            "embedding_service": {
                "status": "healthy",
                "model": embedding_service.model
            },
            "rag_agent": {
                "status": "healthy",
                "model": rag_agent.model
            }
        }
        
        # Check for problems in vector store metrics
        if hasattr(vector_store, 'metrics') and vector_store.metrics["search_errors"] > 3:
            components_status["vector_store"]["status"] = "degraded"
            system_status = "degraded"
            
        # Performance metrics
        metrics = {
            "vector_store": vector_metrics,
            # Add more component metrics here as needed
        }
        
        return {
            "status": system_status,
            "components": components_status,
            "metrics": metrics
        }
    except Exception as e:
        logger.exception(f"Error getting monitoring information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error monitoring system: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint that checks if the API is up
    """
    health_info = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "message": "API is operational",
        "components": {
            "api": "ok"
        }
    }
    
    # Check Ragie if enabled
    if RAGIE_ENABLED:
        try:
            # Create Ragie client and try a simple operation
            ragie_client = RagieClient()
            # Just get document count as a simple check
            docs = ragie_client.get_all_documents()
            health_info["components"]["ragie"] = "ok"
            health_info["components"]["ragie_docs"] = len(docs)
        except Exception as e:
            health_info["components"]["ragie"] = "error"
            health_info["status"] = "degraded"
            health_info["components"]["ragie_error"] = str(e)
    
    return health_info


@app.get("/api/documents", response_model=List[Dict])
async def list_documents():
    """
    List all documents in the system.
    
    Returns:
        List of document information
        
    Raises:
        HTTPException: If there's an error fetching documents
    """
    try:
        # Check if we're using Ragie
        if RAGIE_ENABLED:
            try:
                # Initialize Ragie agent for document operations
                ragie_agent = RagieRAGAgent()
                
                # Get document list from Ragie
                documents = ragie_agent.list_documents()
                
                # Map Ragie document format to a consistent API response
                formatted_documents = []
                
                for doc in documents:
                    # Extract document name from metadata if available
                    doc_name = doc.get("metadata", {}).get("document_name") or doc.get("metadata", {}).get("title") or f"Document {doc['id']}"
                    
                    # Map document info to consistent format
                    formatted_documents.append({
                        "id": doc["id"],
                        "name": doc_name,
                        "status": doc.get("status", "unknown"),
                        "metadata": doc.get("metadata", {})
                    })
                
                return formatted_documents
            except Exception as e:
                logger.exception(f"Error listing documents from Ragie: {e}")
                logger.warning("Returning empty document list due to Ragie API error")
                # Return empty list instead of raising an exception
                return []
        else:
            # Use original document listing if Ragie is not enabled
            # Get all document files from the documents directory
            document_files = list(Path(DOCUMENTS_DIR).glob("*.*"))
            
            # Extract document information
            documents = []
            
            for doc_path in document_files:
                doc_id = doc_path.stem
                status = document_processing_status.get(doc_id, "complete")
                
                # Check if the document has metadata
                metadata = {}
                metadata_path = Path(CHUNKS_DIR) / f"{doc_id}_metadata.json"
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid metadata file for document {doc_id}")
                
                # Get document name from metadata or filename
                doc_name = metadata.get("title") or doc_path.name
                
                documents.append({
                    "id": doc_id,
                    "name": doc_name,
                    "status": status,
                    "metadata": metadata
                })
            
            return documents
    except Exception as e:
        logger.exception(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )


@app.get("/api/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """
    Get the status of a document.
    
    Args:
        document_id: The ID of the document
        
    Returns:
        Dict with document status
        
    Raises:
        HTTPException: If the document is not found
    """
    try:
        # Check if we're using Ragie
        if RAGIE_ENABLED:
            try:
                # Initialize Ragie client
                ragie_client = RagieClient()
                
                # Get document status from Ragie
                status = ragie_client.get_document_status(document_id)
                
                # Map Ragie status to our standardized status
                normalized_status = map_status(status)
                
                # Get appropriate status code based on the status
                status_code = get_document_status_code(normalized_status)
                
                return JSONResponse(
                    status_code=status_code,
                    content={
                        "status": normalized_status,
                        "message": f"Document status: {normalized_status}",
                        "document_id": document_id
                    }
                )
            except Exception as e:
                logger.exception(f"Error checking document status in Ragie: {e}")
                
                # Check if this is a "document not found" error
                if "not found" in str(e).lower():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Document with ID {document_id} not found"
                    )
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Error checking document status: {str(e)}"
                )
        else:
            # Use original document status tracking if Ragie is not enabled
            if document_id not in document_processing_status:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document with ID {document_id} not found"
                )
                
            status = document_processing_status[document_id]
            
            # Map status to standardized format
            normalized_status = map_status(status)
            
            # Get appropriate status code
            status_code = get_document_status_code(normalized_status)
            
            return JSONResponse(
                status_code=status_code,
                content={
                    "status": normalized_status,
                    "message": f"Document status: {normalized_status}",
                    "document_id": document_id
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error checking document status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking document status: {str(e)}"
        )


def get_document_status_code(status: str) -> int:
    """
    Returns the appropriate HTTP status code based on document status.
    
    Args:
        status: The document status string
        
    Returns:
        int: HTTP status code
    """
    # Map status to appropriate HTTP status code
    status_code_map = {
        "error": 500,
        "not_found": 404,
        "processing": 202,  # Accepted but processing not complete
        # All other statuses (complete, indexed, etc) return 200 OK
    }
    
    return status_code_map.get(status, 200)


def map_status(status: str) -> str:
    """
    Maps various status strings to consistent values.
    
    This ensures that both Ragie statuses and internal statuses 
    are presented consistently in API responses.
    
    Args:
        status: The raw status string
        
    Returns:
        str: Normalized status string
    """
    # Map for Ragie statuses
    ragie_status_map = {
        "QUEUED": "processing",
        "PROCESSING": "processing",
        "COMPLETE": "complete",
        "ERROR": "error",
        "FAILED": "error",
        "INDEXED": "complete",
        "INCOMPLETE": "processing"
    }
    
    # Map for internal statuses
    internal_status_map = {
        "queued": "processing",
        "uploading": "processing",
        "processing": "processing",
        "embedding": "processing",
        "indexing": "processing",
        "complete": "complete",
        "error": "error",
        "timeout": "error"
    }
    
    # First check if the status is in uppercase (likely from Ragie)
    if status.upper() == status:
        return ragie_status_map.get(status, status.lower())
    else:
        return internal_status_map.get(status, status)


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the system.
    
    Args:
        document_id: The ID of the document to delete
        
    Returns:
        Dict with delete status
        
    Raises:
        HTTPException: If the document is not found or cannot be deleted
    """
    try:
        # Check if we're using Ragie
        if RAGIE_ENABLED:
            try:
                # Initialize Ragie client
                ragie_client = RagieClient()
                
                # Delete document from Ragie
                result = ragie_client.delete_document(document_id)
                
                # Remove from local status tracking if present
                if document_id in document_processing_status:
                    del document_processing_status[document_id]
                
                return {
                    "status": "success",
                    "message": f"Document {document_id} has been deleted",
                    "document_id": document_id
                }
            except Exception as e:
                logger.exception(f"Error deleting document from Ragie: {e}")
                
                # Check if this is a "document not found" error
                if "not found" in str(e).lower():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Document with ID {document_id} not found"
                    )
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Error deleting document: {str(e)}"
                )
        else:
            # Use original document deletion if Ragie is not enabled
            # Check if document exists in our tracking
            if document_id not in document_processing_status:
                # Document might still exist on disk, let's check
                doc_files = list(Path(DOCUMENTS_DIR).glob(f"{document_id}*"))
                chunk_files = list(Path(CHUNKS_DIR).glob(f"{document_id}*"))
                vector_files = list(Path(VECTORS_DIR).glob(f"{document_id}*"))
                
                if not doc_files and not chunk_files and not vector_files:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Document with ID {document_id} not found"
                    )
            
            # Delete files associated with this document
            success = False
            
            try:
                # Delete document files
                for file_path in Path(DOCUMENTS_DIR).glob(f"{document_id}*"):
                    file_path.unlink()
                    success = True
                
                # Delete chunk files 
                for file_path in Path(CHUNKS_DIR).glob(f"{document_id}*"):
                    file_path.unlink()
                    success = True
                
                # Delete vector files
                for file_path in Path(VECTORS_DIR).glob(f"{document_id}*"):
                    file_path.unlink()
                    success = True
                
                # Remove from tracking if present
                if document_id in document_processing_status:
                    del document_processing_status[document_id]
                
                # Ask vector store to remove document if it exists
                if rag_agent and hasattr(rag_agent.retriever, 'vector_store'):
                    try:
                        rag_agent.retriever.vector_store.delete_document(document_id)
                    except Exception as e:
                        logger.warning(f"Error removing document from vector store: {e}")
                
                return {
                    "status": "success",
                    "message": f"Document {document_id} has been deleted",
                    "document_id": document_id
                }
            except Exception as e:
                logger.exception(f"Error deleting document files: {e}")
                
                if success:
                    # Partial success
                    return {
                        "status": "partial",
                        "message": f"Document {document_id} was partially deleted. Some files may remain.",
                        "document_id": document_id,
                        "error": str(e)
                    }
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Error deleting document: {str(e)}"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in delete_document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error deleting document: {str(e)}"
        )


# Exception handlers for better error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom handler for HTTP exceptions to ensure consistent error response format
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handler for unexpected exceptions with consistent error format and logging
    """
    # Log the unexpected exception
    logger.exception(f"Unexpected error: {exc}")
    
    # Return a structured error response
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"An unexpected error occurred: {str(exc)}",
            "code": 500
        }
    )


# Helper functions for document retrieval and response generation

async def retrieve_documents(query, filter_metadata=None, top_k=None):
    """
    Retrieve documents relevant to the query.
    
    Args:
        query: The user's query string
        filter_metadata: Optional metadata filter criteria
        top_k: Optional number of top chunks to retrieve
        
    Returns:
        List of document chunks matching the query
    """
    if rag_agent is None:
        logger.error("No RAG agent available for document retrieval")
        return []
    
    try:
        # With Ragie-only mode, we need to use a RagieRAGAgent instance for retrieval
        # Create a temporary agent if needed
        ragie_agent = rag_agent if isinstance(rag_agent, RagieRAGAgent) else RagieRAGAgent()
        
        # Use top_k from config if not specified
        if not top_k:
            top_k = 5
            
        # Retrieve documents using Ragie's retrieval method
        retrieval_results = ragie_agent.ragie_client.retrieve(
            query=query,
            filter_metadata=filter_metadata,
            top_k=top_k
        )
        
        logger.info(f"Retrieved {len(retrieval_results['chunks'])} chunks for query: {query}")
        return retrieval_results['chunks']  # Return chunks directly
    except Exception as e:
        logger.exception(f"Error retrieving documents: {e}")
        return []


async def generate_response(query, docs):
    """
    Generate a response to the query using the retrieved documents.
    
    Args:
        query: The user's query string
        docs: The retrieved document chunks
        
    Returns:
        Generated response text
    """
    if rag_agent is None:
        logger.error("No RAG agent available for response generation")
        return "I'm sorry, the RAG system is currently unavailable. Please try again later."
    
    try:
        # With Ragie-only mode, we need to use a RagieRAGAgent instance for generation
        # Create a temporary agent if needed
        ragie_agent = rag_agent if isinstance(rag_agent, RagieRAGAgent) else RagieRAGAgent()
        
        # Create context from docs
        context = ""
        for i, chunk in enumerate(docs):
            text = chunk.get("text", "")
            score = chunk.get("score", 0)
            doc_id = chunk.get("document_id", "unknown")
            
            # Format chunk with source information
            context += f"[Source {i+1}: Document {doc_id}, Relevance: {score:.2f}]\n{text}\n\n"
        
        # Generate a response using the agent
        response = ragie_agent._generate_response(query, context)
        return response
    except Exception as e:
        logger.exception(f"Error generating response: {e}")
        return f"I encountered an error while generating a response: {str(e)}"


@app.get("/api/documents/paginated")
async def list_documents_paginated(
    page: int = Query(1, ge=1, description="Page number, starting from 1"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    filter_status: Optional[str] = Query(None, description="Filter by document status")
):
    """
    List documents with pagination, sorting and filtering.
    
    Args:
        page: Page number to retrieve (starts at 1)
        page_size: Number of documents per page
        sort_by: Field to sort by
        sort_order: Sort order (asc or desc)
        filter_status: Optional status to filter by
        
    Returns:
        Dict with paginated document list and metadata
        
    Raises:
        HTTPException: If there's an error fetching documents
    """
    try:
        # Check if we're using Ragie
        if RAGIE_ENABLED:
            try:
                # Initialize Ragie client
                ragie_client = RagieClient()
                
                # Get all documents from Ragie - we'll handle pagination in memory
                # since Ragie might not have direct pagination support
                all_documents = ragie_client.get_all_documents()
                
                # Calculate total documents and pages
                total_docs = len(all_documents)
                total_pages = (total_docs + page_size - 1) // page_size if total_docs > 0 else 1
                
                # Validate page number
                if page > total_pages and total_pages > 0:
                    page = total_pages
                
                # Filter by status if requested
                if filter_status:
                    normalized_filter = filter_status.upper() if filter_status.upper() == filter_status else filter_status
                    filtered_documents = [
                        doc for doc in all_documents 
                        if doc.get("status", "").upper() == normalized_filter.upper()
                    ]
                    all_documents = filtered_documents
                    
                    # Recalculate totals after filtering
                    total_docs = len(all_documents)
                    total_pages = (total_docs + page_size - 1) // page_size if total_docs > 0 else 1
                
                # Sort documents (default to created_at timestamp)
                # Map field names to Ragie's field names if needed
                sort_field_map = {
                    "created_at": "created_at",
                    "updated_at": "updated_at",
                    "name": "metadata.title",  # Assuming title in metadata
                    "status": "status"
                }
                
                # Get the actual field to sort by
                actual_sort_field = sort_field_map.get(sort_by, "created_at")
                
                # Handle nested fields (e.g., metadata.title)
                if "." in actual_sort_field:
                    parent, child = actual_sort_field.split(".", 1)
                    
                    def get_sort_key(doc):
                        return doc.get(parent, {}).get(child, "")
                else:
                    def get_sort_key(doc):
                        return doc.get(actual_sort_field, "")
                
                # Sort documents
                reverse_sort = sort_order.lower() == "desc"
                sorted_documents = sorted(all_documents, key=get_sort_key, reverse=reverse_sort)
                
                # Get documents for the requested page
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                page_documents = sorted_documents[start_idx:end_idx]
                
                # Format documents for response
                formatted_documents = []
                
                for doc in page_documents:
                    # Extract document name from metadata if available
                    doc_name = doc.get("metadata", {}).get("title") or doc.get("metadata", {}).get("document_name") or f"Document {doc['id']}"
                    
                    # Format document for response
                    formatted_documents.append({
                        "id": doc["id"],
                        "name": doc_name,
                        "status": map_status(doc.get("status", "unknown")),
                        "metadata": doc.get("metadata", {}),
                        "created_at": doc.get("created_at", ""),
                        "updated_at": doc.get("updated_at", "")
                    })
                
                # Return paginated response
                return {
                    "documents": formatted_documents,
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "total_pages": total_pages,
                        "total_documents": total_docs
                    },
                    "filter": {
                        "status": filter_status
                    },
                    "sort": {
                        "field": sort_by,
                        "order": sort_order
                    }
                }
            except Exception as e:
                logger.exception(f"Error listing documents from Ragie: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error listing documents: {str(e)}"
                )
        else:
            # Use original document listing with manual pagination
            try:
                # Get all document files from the documents directory
                document_files = list(Path(DOCUMENTS_DIR).glob("*.*"))
                
                # Extract document information
                all_documents = []
                
                for doc_path in document_files:
                    doc_id = doc_path.stem
                    status = document_processing_status.get(doc_id, "complete")
                    
                    # Map status to standardized format
                    normalized_status = map_status(status)
                    
                    # Check if we should filter by status
                    if filter_status and normalized_status != filter_status:
                        continue
                    
                    # Check if the document has metadata
                    metadata = {}
                    metadata_path = Path(CHUNKS_DIR) / f"{doc_id}_metadata.json"
                    
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, "r") as f:
                                metadata = json.load(f)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid metadata file for document {doc_id}")
                    
                    # Get document name from metadata or filename
                    doc_name = metadata.get("title") or doc_path.name
                    
                    # Get file stats for created/modified times
                    file_stat = doc_path.stat()
                    created_at = datetime.fromtimestamp(file_stat.st_ctime).isoformat()
                    updated_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    
                    all_documents.append({
                        "id": doc_id,
                        "name": doc_name,
                        "status": normalized_status,
                        "metadata": metadata,
                        "created_at": created_at,
                        "updated_at": updated_at
                    })
                
                # Calculate total documents and pages
                total_docs = len(all_documents)
                total_pages = (total_docs + page_size - 1) // page_size if total_docs > 0 else 1
                
                # Validate page number
                if page > total_pages and total_pages > 0:
                    page = total_pages
                
                # Sort documents
                sort_field = sort_by
                reverse_sort = sort_order.lower() == "desc"
                
                sorted_documents = sorted(
                    all_documents, 
                    key=lambda x: x.get(sort_field, ""),
                    reverse=reverse_sort
                )
                
                # Get documents for the requested page
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                page_documents = sorted_documents[start_idx:end_idx]
                
                # Return paginated response
                return {
                    "documents": page_documents,
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "total_pages": total_pages,
                        "total_documents": total_docs
                    },
                    "filter": {
                        "status": filter_status
                    },
                    "sort": {
                        "field": sort_by,
                        "order": sort_order
                    }
                }
            except Exception as e:
                logger.exception(f"Error listing documents with pagination: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error listing documents: {str(e)}"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in list_documents_paginated: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error listing documents: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Render sets this) or use configured API_PORT
    port = int(os.getenv("PORT", API_PORT))
    
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=port,
        log_level="info",
        reload=DEBUG,
        workers=1,
        timeout_keep_alive=120,
    ) 