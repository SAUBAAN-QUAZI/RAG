# RAG UI with Ragie Integration

This is a modern frontend application for interacting with the Ragie RAG (Retrieval Augmented Generation) system. It provides a user-friendly interface for uploading documents, managing your document collection, and asking questions that leverage RAG capabilities.

## Features

- **Document Management**:
  - Upload PDF documents (single or batch)
  - View document status and metadata
  - Delete documents when no longer needed

- **RAG-powered Chat**:
  - Ask questions about your documents
  - View source chunks for transparency
  - Configure retrieval settings (reranking, top-k, etc.)
  - Filter queries by specific documents

- **Advanced Features**:
  - Document metadata support
  - Query timing information
  - Markdown rendering in responses
  - Copy/regenerate responses

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Ragie backend server running

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   cd frontend/rag-ui
   npm install
   # or
   yarn install
   ```

3. Configure the environment:
   Create a `.env.local` file in the project root with:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```
   Replace the URL with your Ragie API endpoint.

4. Run the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

### Document Upload

1. Navigate to the "Document Upload" page
2. Drag and drop PDF files or click to select files
3. Add optional metadata (title, author, description)
4. Click "Upload" and wait for processing to complete
5. View uploaded documents in the list below

### Chat with Documents

1. Navigate to the "Chat" page
2. Use the gear icon to configure retrieval settings
3. Optionally select specific documents to query
4. Type your question and press Enter or click Send
5. View the AI's response with source information
6. Click the dropdown to see source chunks that contributed to the answer

## Integration with Ragie

This frontend communicates with the Ragie backend through a REST API. The integration supports:

- Document upload and processing
- Document listing and management
- RAG queries with configurable parameters
- Source retrieval and reranking

## Configuration Options

### Environment Variables

- `NEXT_PUBLIC_API_URL`: Backend API URL (required)
- `NODE_ENV`: Set to 'production' for production deployment

### Chat Options

- **Reranking**: Enable or disable reranking of retrieved chunks
- **Top-K**: Number of chunks to retrieve (1-10)
- **Show Timings**: Display query processing time information
- **Show Source Documents**: Display source chunks that contributed to the answer

## Development

The project structure follows Next.js conventions:

- `src/components/`: UI components 
- `src/api/`: API client and types
- `src/config/`: Configuration settings
- `src/pages/`: Next.js pages
- `public/`: Static assets

## Troubleshooting

If you encounter issues:

1. Check the Ragie backend server is running
2. Verify API URL in `.env.local` is correct
3. Look for errors in browser console
4. Check network requests in browser developer tools
5. Ensure all dependencies are installed
6. Check backend logs for errors

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Handling Large Document Uploads

The application has been configured to properly handle large document uploads (up to 50MB) with dynamic timeouts:

- Base timeout: 60 seconds for all requests
- Additional time: 30 seconds per MB of file size
- Maximum timeout: 5 minutes (300 seconds)

These values can be adjusted in `src/config.ts` if you experience timeout issues:

```typescript
// Timeout configuration (milliseconds)
baseTimeout: 60000, // 60 seconds base timeout
timeoutPerMb: 30000, // 30 seconds per MB 
maxTimeout: 300000, // Maximum timeout (5 minutes)
```

### Tips for Large Document Uploads

When uploading large documents (over 10MB):

1. Be patient - the server needs time to process the document
2. The UI will show progress and estimated remaining time
3. If you encounter repeated timeouts:
   - Try a smaller document (split large PDFs into smaller parts)
   - Consider increasing the timeouts in `src/config.ts`
   - Check server logs for memory or processing issues

### Ragie Webhook Support

The application supports Ragie webhooks for document processing status updates. To configure webhooks:

1. Set up webhook endpoints in the Ragie dashboard
2. Configure the webhook URL to point to your backend endpoint
3. Implement proper signature validation using the provided utilities in `src/api/webhookUtils.ts`

For development and testing of webhooks, consider using tools like ngrok to expose your local server to the internet.
