"""
FastAPI Application for RAG System
-------------------------------
This module implements a FastAPI application for the RAG system.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag.config import DOCUMENTS_DIR
from rag.retrieval.rag_agent import RAGAgent
from rag.utils import logger

# Initialize the RAG agent
rag_agent = RAGAgent()

# Create the FastAPI application
app = FastAPI(
    title="RAG API",
    description="API for a Retrieval-Augmented Generation system",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request and response models
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
        # Process the query
        answer = rag_agent.query(request.query, filter_dict=request.filters)
        
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """
    Upload and process a document.
    """
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
        # Create a temporary file
        temp_file_path = Path(DOCUMENTS_DIR) / file.filename
        
        # Save the uploaded file
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
            
        # Create metadata
        metadata = {}
        if title:
            metadata["title"] = title
        if author:
            metadata["author"] = author
        if description:
            metadata["description"] = description
            
        # Process the document
        rag_agent.add_document(str(temp_file_path), **metadata)
        
        return JSONResponse(
            status_code=200,
            content={"message": f"Document {file.filename} processed successfully"},
        )
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    # Run the API server
    uvicorn.run("app.main:app", host=host, port=port, reload=True) 