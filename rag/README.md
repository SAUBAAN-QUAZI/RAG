# RAG System with Ragie.ai Integration

This is a Retrieval-Augmented Generation (RAG) system with full integration support for Ragie.ai. The system allows you to upload documents, query them using natural language, and generate context-aware responses.

## Features

- **Document Management**: Upload, list, and delete documents with Ragie.ai
- **Advanced Document Processing**: Support for fast and hi-res document processing modes
- **Document Chunking**: Automatic document chunking for efficient retrieval
- **Document Content Access**: Retrieve document content, chunks, and summaries
- **Semantic Search**: Perform semantic search on documents with reranking
- **Partitioning**: Support for document partitioning to organize your knowledge base
- **Rich Metadata**: Add and filter by metadata for more targeted retrieval

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Copy the environment template and configure it:

```bash
cp .env.template .env
```

3. Edit the `.env` file to add your API keys and configure the system:

```properties
# Required API keys
OPENAI_API_KEY=your_openai_api_key_here
RAGIE_API_KEY=your_ragie_api_key_here

# Enable Ragie integration
USE_RAGIE=true
```

## Usage

### Using the Ragie.ai Integration

The `RagieClient` class provides a comprehensive interface to the Ragie.ai API:

```python
from rag.integrations import create_ragie_client

# Create a client instance
client = create_ragie_client()

# Upload a document
result = client.upload_document(
    file_path="path/to/document.pdf",
    metadata={"source": "example", "author": "John Doe"},
    mode="fast"  # or "hi_res" for better image and table extraction
)

# Wait for document processing to complete
document_id = result["id"]
client.wait_for_document_ready(document_id)

# List all documents
documents = client.get_all_documents()

# Get document chunks
chunks = client.get_document_chunks(document_id)

# Get document summary
summary = client.get_document_summary(document_id)

# Perform retrieval
results = client.retrieve(
    query="What is the main topic of the document?",
    document_ids=[document_id],
    rerank=True,
    top_k=5
)

# Delete a document
client.delete_document(document_id)
```

### Example Script

You can try out the Ragie.ai integration using the provided example script:

```bash
# Upload a document
python examples/ragie_example.py upload path/to/document.pdf

# List all documents
python examples/ragie_example.py list

# Get document chunks
python examples/ragie_example.py chunks document_id

# Get document summary
python examples/ragie_example.py summary document_id

# Perform retrieval
python examples/ragie_example.py retrieve "What is the main topic of the document?"

# Delete a document
python examples/ragie_example.py delete document_id
```

## Configuration Options

The Ragie.ai integration can be configured using the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_RAGIE` | Enable Ragie.ai integration | `false` |
| `RAGIE_API_KEY` | Your Ragie.ai API key | - |
| `RAGIE_DEFAULT_PARTITION` | Default partition for documents | `default` |
| `RAGIE_PROCESS_MODE` | Document processing mode (`fast` or `hi_res`) | `fast` |
| `RAGIE_WAIT_FOR_READY` | Wait for document to be fully ready | `true` |
| `RAGIE_ACCEPT_INDEXED` | Accept "indexed" state as ready | `true` |
| `RAGIE_TIMEOUT` | Timeout for document processing (seconds) | `300` |
| `RAGIE_REQUEST_TIMEOUT` | Timeout for API requests (seconds) | `30` |
| `RAGIE_WEBHOOK_SECRET` | Secret for webhook authentication | - |

## Advanced Features

### Document Processing Modes

- **Fast Mode**: Quickly extracts text from documents (default)
- **Hi-Res Mode**: Extracts text, images, and tables with higher fidelity (slower but more accurate)

### Partitioning

Partitions allow you to organize your documents into separate collections:

```python
# Upload to a specific partition
client.upload_document(
    file_path="path/to/document.pdf",
    partition="project_a"
)

# List documents in a partition
client.get_all_documents(partition="project_a")

# Query documents in a partition
client.retrieve(
    query="What is the main topic?",
    partition="project_a"
)
```

### Raw Text Upload

Upload raw text without a file:

```python
client.upload_document_raw(
    data="This is some text to upload",
    name="example-text.txt",
    metadata={"source": "example"}
)
```

### URL-based Upload

Upload a document from a URL:

```python
client.upload_document_from_url(
    url="https://example.com/document.pdf",
    name="example-document.pdf",
    metadata={"source": "web"}
)
```

## Troubleshooting

If you encounter issues with the Ragie.ai integration:

1. Ensure you have a valid API key set in the `.env` file
2. Check that the `ragie` Python package is installed
3. Verify that `USE_RAGIE` is set to `true` in your `.env` file
4. Check the logs for detailed error messages

For more detailed information, refer to the Ragie.ai API documentation. 