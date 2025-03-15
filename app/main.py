"""
FastAPI Application
---------------
This module implements the RESTful API for the RAG system.
"""

import os
import secrets
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.config import (
    API_HOST, API_PORT, DEBUG, ALLOW_CORS, CORS_ORIGINS, DOCUMENTS_DIR, 
    OPENAI_API_KEY, DATA_DIR, EMBEDDING_MODEL, MAX_RESPONSE_TOKENS
)
from rag.retrieval.rag_agent import RAGAgent
from rag.utils import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="API for the Retrieval-Augmented Generation system",
    version="0.1.0",
    debug=DEBUG,
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

# Log important startup information
logger.info(f"Starting RAG API on {API_HOST}:{API_PORT}")
logger.info(f"DEBUG mode: {DEBUG}")
logger.info(f"CORS enabled: {ALLOW_CORS}, origins: {CORS_ORIGINS}")
logger.info(f"Using vector DB type: {os.getenv('VECTOR_DB_TYPE', 'chroma')}")
logger.info(f"Storage directories: Documents={DOCUMENTS_DIR}")

# Initialize RAG agent
try:
    rag_agent = RAGAgent()
    logger.info("Successfully initialized RAG agent")
except Exception as e:
    logger.error(f"Failed to initialize RAG agent: {e}")
    logger.exception("Detailed error:")
    # We'll still define rag_agent to avoid NameErrors, but mark it as failed
    rag_agent = None

# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes

# Document processing tracker
document_processing_status = {}

# Function to process document in the background
def process_document_task(file_path: str, doc_id: str, metadata: dict):
    """
    Process a document in the background
    """
    try:
        logger.info(f"Starting background processing of document {doc_id}")
        document_processing_status[doc_id] = "processing"
        
        # Process the document
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
        file_result["details"].update({
            "error": f"Error processing document: {str(e)}"
        })
        
        # Clean up the temporary file in case of processing error
        if os.path.exists(file_path):
            os.unlink(file_path)
            
        logger.exception(f"Error processing batch document: {e}")


class QueryRequest(BaseModel):
    """
    Request model for a query.
    """
    query: str = Field(..., description="Query text")
    filters: Optional[Dict] = Field(None, description="Metadata filters")


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


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query using the RAG system.
    """
    # Check if RAG agent was successfully initialized
    if rag_agent is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system is not available. Check server logs for initialization errors."
        )
        
    try:
        # Get response from RAG agent
        response = rag_agent.query(request.query, request.filters)
        
        # Handle response that could be either string or dictionary
        if isinstance(response, dict):
            logger.info("Received structured response from RAG agent, extracting answer field")
            answer = response.get("answer", "")
            # Log additional information that won't be returned to the client
            if "sources" in response:
                logger.info(f"Sources: {response['sources']}")
            if "confidence" in response:
                logger.info(f"Confidence: {response['confidence']}")
        else:
            # Response is already a string
            answer = response
            
        return {"answer": answer}
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """
    Upload a document to the RAG system.
    """
    # Check if RAG agent was successfully initialized
    if rag_agent is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system is not available. Check server logs for initialization errors."
        )
        
    try:
        # Check file size
        if file.size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE/1024/1024}MB"
            )
        
        # Check file type (only allow PDF for now)
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only PDF files are supported."
            )
        
        # Log file info
        logger.info(f"Received file: {file.filename}, size: {file.size}, content-type: {file.content_type}")
        
        # Create a temporary file for storage
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=DOCUMENTS_DIR) as temp_file:
            # Read the file in chunks to avoid memory issues with large files
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Get document ID for tracking
        doc_id = os.path.basename(temp_path)
        
        # Initialize status in the tracker
        document_processing_status[doc_id] = "uploading"
        
        # Prepare metadata
        metadata = {
            "title": title or file.filename,
            "author": author or "Unknown",
            "description": description or "",
            "original_filename": file.filename,
            "content_type": file.content_type,
            "file_size": file.size,
            "processing_status": "processing",  # Add processing status
            "upload_time": datetime.now().isoformat(),
            "source_document_id": doc_id,  # Add source document ID to help with existence checking
        }
        
        # Schedule the background task to process the document
        background_tasks.add_task(process_document_task, temp_path, doc_id, metadata)
        
        # Return an initial response to indicate the upload was successful
        # and processing has started
        return JSONResponse(
            status_code=202,  # 202 Accepted to indicate processing
            content={
                "message": f"Document '{file.filename}' uploaded and is being processed.",
                "file_path": temp_path,
                "document_id": doc_id,
                "status": "processing",  # This status reflects that the upload succeeded and processing has started
                "metadata": metadata,
            },
        )
            
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.exception(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    Upload multiple documents to the RAG system.
    
    Documents are processed in the background. If any document fails, 
    the endpoint will continue processing the remaining documents.
    """
    # Check if RAG agent was successfully initialized
    if rag_agent is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system is not available. Check server logs for initialization errors."
        )
    
    # Track processing results
    results = []
    successful_count = 0
    failed_count = 0
    
    # Check if there are any files to process
    if not files or len(files) == 0:
        return JSONResponse(
            status_code=400,
            content={"message": "No files provided for upload."}
        )
    
    # Check if too many files
    if len(files) > 10:
        return JSONResponse(
            status_code=400,
            content={"message": f"Too many files. Maximum allowed is 10, received {len(files)}."}
        )
    
    # Process each file
    for idx, file in enumerate(files):
        file_result = {
            "filename": file.filename,
            "status": "pending",
            "details": {}
        }
        
        try:
            # Check file size
            if file.size > MAX_FILE_SIZE:
                file_result["status"] = "error"
                file_result["details"] = {
                    "error": f"File size exceeds maximum allowed size of {MAX_FILE_SIZE/1024/1024}MB"
                }
                failed_count += 1
                results.append(file_result)
                continue
            
            # Check file type
            if not file.filename.lower().endswith(".pdf"):
                file_result["status"] = "error"
                file_result["details"] = {
                    "error": "Unsupported file type. Only PDF files are supported."
                }
                failed_count += 1
                results.append(file_result)
                continue
            
            # Log file info
            logger.info(f"Processing batch file {idx+1}/{len(files)}: {file.filename}, size: {file.size}")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=DOCUMENTS_DIR) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name
            
            # Get document ID for tracking
            doc_id = os.path.basename(temp_path)
            
            # Initialize status in the tracker
            document_processing_status[doc_id] = "uploading"
            
            # Prepare metadata
            doc_title = f"{title_prefix + ' - ' if title_prefix else ''}{file.filename}"
            metadata = {
                "title": doc_title,
                "author": author or "Unknown",
                "description": description or "",
                "original_filename": file.filename,
                "content_type": file.content_type,
                "file_size": file.size,
                "batch_upload": True,
                "batch_index": idx + 1,
                "processing_status": "processing",
                "upload_time": datetime.now().isoformat(),
                "source_document_id": doc_id,
            }
            
            # Update file result for successful upload (but still processing)
            file_result["status"] = "processing"
            file_result["details"] = {
                "message": f"Document '{file.filename}' uploaded and is being processed.",
                "file_path": temp_path,
                "document_id": doc_id,
            }
            successful_count += 1
            
            # Add file result to tracking
            results.append(file_result)
            
            # Schedule background processing for this document
            background_tasks.add_task(
                process_batch_document_task,
                temp_path, 
                doc_id, 
                metadata,
                file_result
            )
                
        except Exception as e:
            # File handling error
            file_result["status"] = "error"
            file_result["details"] = {
                "error": f"Error handling file: {str(e)}"
            }
            logger.exception(f"Error handling batch file {file.filename}: {e}")
            failed_count += 1
            
            # Add result to tracking
            results.append(file_result)
    
    # Return overall results
    return JSONResponse(
        status_code=202,  # 202 Accepted to indicate processing in progress
        content={
            "message": f"Batch upload initiated. {successful_count} uploaded successfully, {failed_count} failed. Processing will continue in the background.",
            "successful_count": successful_count,
            "failed_count": failed_count,
            "total_count": len(files),
            "results": results
        }
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
    Health check endpoint for monitoring.
    """
    # Check if OpenAI API key is available
    api_key_valid = bool(OPENAI_API_KEY) and OPENAI_API_KEY != "your_openai_api_key"
    
    # Check if directories are writable
    data_dir_writable = os.access(DATA_DIR, os.W_OK)
    
    # Check embedding model
    embedding_model_info = {
        "name": EMBEDDING_MODEL,
        "valid": EMBEDDING_MODEL in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
    }
    
    # Determine overall health status
    is_healthy = api_key_valid and data_dir_writable and embedding_model_info["valid"]
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "checks": {
            "api_key": api_key_valid,
            "data_directory": data_dir_writable,
            "embedding_model": embedding_model_info
        },
        "message": "Welcome to the RAG API. System is operational."
    }


# Additional error handling for deployment
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
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