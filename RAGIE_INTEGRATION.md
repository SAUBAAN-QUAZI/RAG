# Ragie Integration

This document describes the integration of Ragie's managed RAG service into our system.

## Overview

Ragie is a fully managed Retrieval-Augmented Generation service that handles document processing, embedding, and retrieval. By integrating Ragie, we simplify our architecture and gain access to advanced features like:

- Automatic document processing with hi_res mode for images and tables
- Managed vector storage and embedding
- Hybrid retrieval (semantic + keyword search)
- Document summarization 
- Support for 30+ document formats
- Webhook integration for real-time status updates

## Architecture

The integration uses a facade pattern where our existing API interfaces remain largely the same, but the underlying implementation delegates to Ragie's services:

```
┌─────────────┐      ┌─────────────┐      ┌──────────────┐
│   API Layer │──────▶ RagieRAGAgent│──────▶  RagieClient │
└─────────────┘      └─────────────┘      └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │  Ragie API   │
                                          └──────────────┘
```

## Components

1. **RagieClient** (`rag/integrations/ragie.py`): Handles direct communication with Ragie API
2. **RagieRAGAgent** (`rag/retrieval/ragie_agent.py`): Implements our RAG pipeline using Ragie 
3. **API Layer** (`app/main.py`): API endpoints that remain compatible with existing clients

## Configuration

To enable Ragie integration:

1. Add your Ragie API key to the `.env` file:
   ```
   RAGIE_API_KEY=your_ragie_api_key_here
   USE_RAGIE=True
   RAGIE_WEBHOOK_SECRET=your_webhook_secret_here  # Optional, for webhook verification
   ```

2. Install the required package:
   ```
   pip install ragie>=2.0.0
   ```

3. SDK Initialization:
   ```python
   from ragie import Ragie
   
   # Using API key directly
   client = Ragie(auth="YOUR_API_KEY")
   
   # Or from environment variable
   from dotenv import load_dotenv
   import os
   
   load_dotenv()
   client = Ragie(auth=os.environ.get("RAGIE_API_KEY"))
   ```

## SDK Functionality

### Document Processing

Documents uploaded through our API are sent to Ragie for processing. Ragie handles:
- PDF parsing 
- Chunking
- Embedding generation
- Vector storage
- Table and image extraction (in hi_res mode)

#### Single Document Upload
```python
with open("contract.pdf", "rb") as f:
    response = client.documents.create(
        request={
            "file": {
                "file_name": "contract.pdf",
                "content": f.read()
            },
            "metadata": {
                "department": "legal",
                "effective_date": "2025-03-01"
            }
        }
    )
print(f"Document ID: {response.id}")
```

#### Batch Document Processing
```python
batch = [
    {"display_name": "report.docx", "blob": open("report.docx", "rb")},
    {"display_name": "meeting.mp3", "blob": open("meeting.mp3", "rb")}
]
for doc in batch:
    with open(doc["blob"], "rb") as f:
        client.documents.create(
            request={
                "file": {
                    "file_name": doc["display_name"],
                    "content": f.read()
                }
            }
        )
```

### Retrieval

Queries are processed by:
1. Preprocessing the query (optional)
2. Sending to Ragie's retrieval API
3. Formatting the results
4. Generating a response using the retrieved context

#### Retrieval with Filters
```python
results = client.retrievals.create(
    request={
        "query": "Q4 financial projections",
        "filter": "{\"department\":\"finance\"}",
        "rerank": True,
        "top_k": 5
    }
)

for chunk in results.chunks:
    print(f"Relevance: {chunk.score:.2f} | {chunk.text[:100]}...")
```

### Document Management

The integration supports:
- Listing all documents
- Checking document status
- Deleting documents

#### Listing Documents with Pagination
```python
# Get first page of documents
response = client.documents.list()
documents = []

# Process current page
for doc in response.documents:
    documents.append({
        "id": doc.id,
        "status": doc.status,
        "metadata": doc.metadata
    })

# If there are more pages, continue fetching
next_page = response.next()
while next_page:
    for doc in next_page.documents:
        documents.append({
            "id": doc.id,
            "status": doc.status,
            "metadata": doc.metadata
        })
    next_page = next_page.next()
```

### Webhook Implementation

Ragie provides webhooks for real-time status updates. We've implemented webhook handlers in `app/webhooks.py`:

```python
@webhooks_router.post("/ragie")
async def ragie_webhook(request: Request):
    """Webhook handler for Ragie events"""
    body = await request.body()
    
    # Verify signature
    signature = request.headers.get("X-Signature")
    if RAGIE_WEBHOOK_SECRET and not verify_signature(body, signature):
        return Response(status_code=401, content="Invalid signature")
    
    event = json.loads(body)
    event_type = event.get("type")
    
    # Process event
    if event_type == "document_status_updated":
        document_id = event.get("document_id")
        status = event.get("status")
        # Update local status tracking
        logger.info(f"Document {document_id} status: {status}")
    
    return {"status": "success"}
```

To configure webhooks in Ragie:
1. Go to the Ragie dashboard
2. Navigate to the Webhooks section
3. Add a new webhook with your endpoint URL (e.g., `https://yourdomain.com/webhooks/ragie`)
4. Subscribe to events like `document_status_updated`
5. Use a webhook secret for security

## Advanced Features

### Hybrid Search Configuration
```python
retrieval_params = {
    "query": "patent filings",
    "vector_weight": 0.7,  # Weight for semantic search
    "keyword_weight": 0.3,  # Weight for keyword search
    "summary_index": True   # Use document summaries if available
}
```

### Metadata Filtering
```python
filters = {
    "numeric": {"page_count": {"gt": 10}},
    "categorical": {"document_type": ["contract", "nda"]},
    "temporal": {"upload_date": {"after": "2025-01-01"}}
}
# Convert to JSON string for Ragie API
import json
filter_string = json.dumps(filters)
```

### Error Handling
```python
from ragie.models import SDKError

try:
    document = client.documents.create(request={...})
except SDKError as e:
    print(f"Error: {e}")
```

### External Data Connectors
Ragie offers 20+ pre-built connectors for data sources:
```python
client.connections.create(
    source_type="google_drive",  # Or "notion", "sharepoint", etc.
    chunk_strategy="hierarchical",  # Alternative: "fixed_size"
    embedding_model="text-embedding-3-large"
)
```

## API Changes

The API endpoints are backward compatible but support these additional features:

- `/api/query`: Added support for Ragie-specific params like `rerank` and `metadata_filter`
- `/api/documents/{id}/status`: Now checks status from Ragie
- `/api/documents`: Lists documents from Ragie
- `/webhooks/ragie`: New endpoint for receiving Ragie webhook events

## Migration Path from Custom RAG

When migrating from a custom RAG system to Ragie:

1. **Replace document processing** components with Ragie's document creation API
2. **Remove vector storage** infrastructure as Ragie provides its own
3. **Update retrieval logic** to use Ragie's retrieval API
4. **Implement webhooks** for real-time status updates
5. **Keep your API layer** but update it to delegate to Ragie

## Limitations

- Document processing options are limited to what Ragie supports
- Customization of chunking and embedding is handled by Ragie
- Metadata filtering syntax must follow Ragie's JSON format

## Dependencies

- `ragie`: The Ragie Python SDK 
- `openai`: For response generation from retrieved context 