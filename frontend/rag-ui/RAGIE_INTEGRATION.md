# Ragie Frontend Integration

This document outlines the changes made to integrate the frontend application with Ragie, our RAG (Retrieval Augmented Generation) service.

## Overview of Changes

1. **API Client Updates**
   - Modified `ragApi.ts` to support Ragie's API endpoints and parameters
   - Added support for document-specific queries and filtering
   - Enhanced error handling and retry logic
   - Added timings and source chunk support in responses

2. **Document Upload Component**
   - Updated to work with Ragie's document processing endpoints
   - Added support for document deletion and listing
   - Improved metadata handling and progress tracking
   - Added document details modal with metadata inspection

3. **Chat Component**
   - Added configurable retrieval parameters (reranking, top-k)
   - Implemented document filtering for targeted queries
   - Added source chunk display with collapsible UI
   - Added query timing information display

4. **UI and UX Improvements**
   - Switched to Chakra UI for consistent design
   - Added toast notifications for better user feedback
   - Implemented markdown rendering for chat responses
   - Added copy and regenerate functionality for chat messages

5. **Configuration Updates**
   - Added environment variables for Ragie-specific settings
   - Implemented configuration helpers for better type safety
   - Added feature flags for conditional feature enablement

## API Changes

### Query Endpoint

**Previous:**
```typescript
interface QueryRequest {
  query: string;
  filters?: Record<string, unknown>;
}

interface QueryResponse {
  answer: string;
}
```

**New (Ragie):**
```typescript
interface QueryRequest {
  query: string;
  document_ids?: string[];
  metadata_filter?: Record<string, any>;
  rerank?: boolean;
  top_k?: number;
  show_timings?: boolean;
}

interface QueryResponse {
  query: string;
  response: string;
  chunks?: Array<{
    text: string;
    score: number;
    metadata: Record<string, any>;
    document_id: string;
  }>;
  document_ids?: string[];
  timings?: Record<string, number>;
}
```

### Document Operations

Added new endpoints for document management:
- `GET /api/documents` - List all documents
- `DELETE /api/documents/{id}` - Delete a document
- `GET /api/documents/{id}/status` - Check document status

## New Features

### Document Management

- View all uploaded documents with status and metadata
- Delete documents that are no longer needed
- View detailed document information

### Enhanced RAG Capabilities

- Filter queries by specific documents
- Configure retrieval parameters (rerank, top-k)
- View source chunks that contributed to answers
- See query timing information for performance analysis

### Improved User Experience

- Better error handling with detailed messages
- Progress tracking for document uploads
- Markdown rendering in chat responses
- Copy and regenerate functionality for responses

## Configuration

The integration uses environment variables for configuration:

```
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# File Upload Configuration
NEXT_PUBLIC_MAX_FILE_SIZE=52428800  # 50MB
NEXT_PUBLIC_MAX_BATCH_SIZE=5  # Max 5 files per batch

# Default RAG Settings
NEXT_PUBLIC_DEFAULT_RERANK=true
NEXT_PUBLIC_DEFAULT_TOP_K=3
```

## Dependencies Added

- `@chakra-ui/react` - UI component library
- `@emotion/react` and `@emotion/styled` - Required for Chakra UI
- `framer-motion` - Required for Chakra UI animations
- `react-markdown` - For rendering markdown in chat responses
- `react-icons` - For icon components

## Testing the Integration

1. Start the Ragie backend server
2. Set the correct `NEXT_PUBLIC_API_URL` in `.env.local`
3. Start the frontend: `npm run dev`
4. Upload a document via the Document Upload page
5. Navigate to the Chat page and ask questions about your document

## Known Limitations

1. Only PDF documents are currently supported
2. Maximum file size is limited to 50MB
3. Batch uploads are limited to 5 files at once
4. Document deletion is permanent and cannot be undone 