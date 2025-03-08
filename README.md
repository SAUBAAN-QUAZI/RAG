# RAG System

A Retrieval-Augmented Generation (RAG) system that combines the power of large language models with document retrieval capabilities.

## Setup Instructions

### Prerequisites
- Python 3.11.7
- OpenAI API key
- Git

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RAG.git
cd RAG
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following content:
```
OPENAI_API_KEY=your_openai_api_key
```

5. Download additional resources (if needed):
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Project Structure

```
RAG/
├── app/                   # FastAPI application
│   ├── api/               # API endpoints
│   ├── core/              # Core functionality
│   └── models/            # Pydantic models
├── rag/                   # RAG system components
│   ├── document_processing/  # Document processing pipeline
│   ├── embedding/         # Embedding service
│   ├── retrieval/         # Retrieval system
│   └── vector_store/      # Vector storage
├── data/                  # Data storage
│   ├── documents/         # Raw documents
│   ├── chunks/            # Processed chunks
│   └── vectors/           # Vector database
├── tests/                 # Test cases
├── .env                   # Environment variables
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Usage

### Starting the API Server

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000.

API Documentation will be available at http://localhost:8000/docs.

### Adding Documents

Documents can be added through the API endpoint or using the document processing utilities directly:

```python
from rag.document_processing import process_document

# Process a document
process_document("path/to/document.pdf")
```

### Querying the RAG System

```python
from rag.retrieval import query_rag

# Query the RAG system
response = query_rag("Your question here?")
print(response)
```

## Development Roadmap

Follow the development phases outlined in the [RAG Implementation Plan](RAG_Implementation_Plan.md).

## License

[MIT License](LICENSE) 