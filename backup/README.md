# Backup of Original RAG System Components

This directory contains backups of the original RAG system components that were replaced by Ragie integration.

## Contents

- `document_processing/`: Original document processing components
- `document/`: Original document models and utils
- `embedding/`: Original embedding services
- `vector_store/`: Original vector store implementations
- `retrieval/`: Original retriever implementation

These components have been superseded by the Ragie integration, which provides managed document processing, embedding, vector storage, and retrieval services.

## Restoration

If you need to revert to the original implementation:
1. Copy the folders back to their original locations in the `rag/` directory
2. Set `USE_RAGIE=False` in your `.env` file

## Migration Date

This backup was created when the system was migrated to use Ragie exclusively. 