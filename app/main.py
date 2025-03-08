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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.config import API_HOST, API_PORT, DEBUG, ALLOW_CORS, CORS_ORIGINS, DOCUMENTS_DIR
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

# Initialize RAG agent
rag_agent = RAGAgent()

# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes


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
    Root endpoint.
    """
    return {"message": "Welcome to the RAG API"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query using the RAG system.
    """
    try:
        answer = rag_agent.query(request.query, request.filters)
        return {"answer": answer}
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """
    Upload a document to the RAG system.
    """
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
        
        # Prepare metadata
        metadata = {
            "title": title or file.filename,
            "author": author or "Unknown",
            "description": description or "",
            "original_filename": file.filename,
            "content_type": file.content_type,
            "file_size": file.size,
        }
        
        # Process document
        try:
            rag_agent.add_document(temp_path, **metadata)
            
            # Return success response
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"Document '{file.filename}' uploaded and processed successfully.",
                    "file_path": temp_path,
                    "metadata": metadata,
                },
            )
        except Exception as e:
            # Clean up the temporary file in case of processing error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            logger.exception(f"Error processing document: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
            
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.exception(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring", response_model=MonitoringResponse)
async def monitor_system():
    """
    Get monitoring information about the RAG system.
    
    Returns detailed status and metrics about all components.
    """
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
    Health check endpoint.
    """
    return {"status": "ok", "version": "0.1.0"}


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
    
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        log_level="info",
        reload=DEBUG,
        workers=1,
        timeout_keep_alive=120,
    ) 