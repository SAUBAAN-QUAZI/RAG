# Ragie.ai API Reference

This document provides reference information for the Ragie.ai API endpoints and parameters. Below is a comparison between the Ragie.ai API and our current implementation.

## Document Management Endpoints

### Create Document
- **Method**: POST
- **Endpoint**: `https://api.ragie.ai/documents`
- **Description**: Upload a document file for ingestion and indexing.

#### Document Processing Pipeline
On ingest, documents progress through these states:
1. `pending`
2. `partitioning`
3. `partitioned`
4. `refined`
5. `chunked`
6. `indexed`
7. `summary_indexed`
8. `keyword_indexed`
9. `ready`
10. `failed` (if errors occur)

> **Note**: Documents are available for retrieval in the `indexed` state, but summaries are only available in `summary_indexed` or `ready` states.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Binary file to upload (supported formats: .eml, .html, .json, .md, .msg, .rst, .rtf, .txt, .xml, .png, .webp, .jpg, .jpeg, .tiff, .bmp, .heic, .csv, .doc, .docx, .epub, .odt, .pdf, .ppt, .pptx, .tsv, .xlsx, .xls) |
| `mode` | String | No | Partition strategy: `fast` (default) or `hi_res`. Hi-res extracts images and tables, but is slower. |
| `metadata` | Object | No | Document metadata as key-value pairs |
| `external_id` | String | No | Optional identifier for document |
| `name` | String | No | Optional document name (defaults to filename) |
| `partition` | String | No | Optional partition identifier |

#### Responses
- `201`: Successful response
- `400`: Bad request
- `401`: Unauthorized
- `402`: Payment required
- `422`: Validation error
- `429`: Too many requests

### Create Document Raw
- **Method**: POST
- **Endpoint**: `https://api.ragie.ai/documents/raw`
- **Description**: Ingest a document as raw text.

### Create Document From URL
- **Method**: POST
- **Endpoint**: `https://api.ragie.ai/documents/url`
- **Description**: Ingest a document from a publicly accessible URL.

## Document Retrieval Endpoints

### Get Document Chunks
- **Method**: GET
- **Endpoint**: `https://api.ragie.ai/documents/{document_id}/chunks`
- **Description**: List all document chunks sorted by index.

### Get Document Chunk
- **Method**: GET
- **Endpoint**: `https://api.ragie.ai/documents/{document_id}/chunks/{chunk_id}`
- **Description**: Get a specific document chunk by ID.

### Get Document Content
- **Method**: GET
- **Endpoint**: `https://api.ragie.ai/documents/{document_id}/content`
- **Description**: Get the raw text content of a document.

### Get Document Source
- **Method**: GET
- **Endpoint**: `https://api.ragie.ai/documents/{document_id}/source`
- **Description**: Get the original source file of a document.

### Get Document Summary
- **Method**: GET
- **Endpoint**: `https://api.ragie.ai/documents/{document_id}/summary`
- **Description**: Get an LLM-generated summary of the document.

## Comparison with Current Implementation

Our current implementation differs significantly from the Ragie.ai API:

### Base URL
- **Ragie.ai**: `https://api.ragie.ai`
- **Current**: Custom backend URL from config (defaults to `http://localhost:8000`)

### API Structure
- **Ragie.ai**: Direct endpoints like `/documents`
- **Current**: Prefixed endpoints like `/api/documents` 

### Document Upload
- **Ragie.ai**: Separate endpoints for different upload types (`/documents`, `/documents/raw`, `/documents/url`)
- **Current**: Single file upload endpoint `/api/documents` and batch upload via `/api/documents/batch`

### Document Status
- **Ragie.ai**: Detailed status progression (pending → partitioning → partitioned → refined → chunked → indexed → summary_indexed → keyword_indexed → ready)
- **Current**: Simpler status tracking with fewer states

### Additional Features
- **Ragie.ai**: Rich metadata support, partition feature, hi-res/fast ingestion modes
- **Current**: Basic metadata support with limited partition capabilities

## Migration Considerations

To align our implementation with the Ragie.ai API:

1. Update endpoint paths to remove the `/api` prefix
2. Implement additional document status states
3. Add support for the different document ingestion endpoints
4. Implement the document retrieval endpoints (chunks, content, source, summary)
5. Add support for partitions and ingestion modes

## Code Changes Required

The main files that would need updates:

1. `ragApi.ts`: Update API endpoint paths and add new functionality
2. `DocumentUpload.tsx`: Update to support different ingestion modes and better status tracking
3. Add new components for document content retrieval, chunk viewing, and summary display 