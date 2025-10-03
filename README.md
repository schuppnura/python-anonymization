# Anonymization RAG System

A privacy-focused document processing and retrieval system using FAISS for vector storage and local AI models through Ollama.

## Features

- Privacy-First: Document anonymization with optional LLM-based privacy filtering
- Vector Search: High-performance semantic search using FAISS
- Local AI: Uses Ollama for embeddings and privacy filtering
- Multi-Format: Supports PDF and DOCX document processing
- RAG Pipeline: Complete ingestion and retrieval workflow
- Modular Design: Clean separation of concerns across components

## Architecture

### Core Components

- Document Loader: Extracts text from PDF and DOCX files
- Chunking: Intelligent text segmentation with overlap
- Embedding: Vector generation using local Ollama models
- FAISS Store: Persistent vector index storage
- Privacy Filter: Optional anonymization using LLM
- Query Engine: Semantic search and retrieval

### Key Design Patterns

- **Pipeline Architecture**: Clear data flow from ingestion to retrieval
- **Configuration-Driven**: JSON-based configuration management
- **Error Resilience**: Comprehensive error handling throughout
- **Modular Components**: Easily extensible and testable design

## Dependencies

- `faiss-cpu`: Vector similarity search
- `numpy`: Numerical computing
- `ollama`: Local AI model integration
- `pypdf`: PDF document processing
- `python-docx`: DOCX document processing

### Development Dependencies

- `black`: Code formatting
- `ruff`: Fast Python linting
- `pytest`: Testing framework
- `mypy`: Static type checking

## Setup

### Prerequisites

- Python 3.11+: Required for modern type hints
- UV Package Manager: Fast Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- Ollama: Local AI model server ([ollama.ai](https://ollama.ai))

### Installation

```bash
# Clone or navigate to the project directory
cd Anonymization

# Install dependencies with UV (uses external virtualenv from .env configuration)
uv sync

# Install development dependencies
uv sync --group dev

# Note: Virtual environment is managed externally at ~/.local/share/virtualenvs/Anonymization
# UV automatically uses it when configured via .env file
```

### Required Models

Install the necessary Ollama models:

```bash
# Embedding model (choose one)
ollama pull mxbai-embed-large
# or
ollama pull bge-large

# Optional: LLM for privacy filtering
ollama pull llama3.1:8b-instruct-q4_K_M
```

## Usage

### Configuration

Create or modify `config.json`:

```json
{
  "index_path": "data/faiss.index",
  "metadata_path": "data/metadata.json",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "embedding_model": "mxbai-embed-large",
  "use_llm_privacy_filter": false
}
```

### Document Ingestion

```bash
# Ingest a document into the FAISS index
uv run main.py ingest path/to/document.pdf

# Ingest with custom config
uv run main.py --config my-config.json ingest document.docx
```

### Querying

```bash
# Search for relevant chunks
uv run main.py query "What is the main topic?"

# Get more results
uv run main.py query "specific question" --k 10
```

### Development Commands

```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking
uv run mypy .

# Run tests
uv run pytest

# Run all development tasks
make dev-check  # if Makefile exists
```

## Project Structure

```
Anonymization/
├── main.py              # CLI entry point
├── query.py             # Search and retrieval
├── document_loader.py   # File parsing utilities
├── anonymiser.py        # Privacy filtering
├── config.json          # Runtime configuration
├── pyproject.toml       # UV dependency management
├── requirements.txt     # Pip fallback dependencies
├── uv.lock              # UV dependency lock file
├── Makefile             # Development automation
├── .env                 # Environment configuration
├── .python-version      # Python version specification
├── tests/               # Test suite
│   ├── conftest.py      # Test configuration
│   ├── test_anonymiser.py # Anonymization tests
│   ├── test_chunking.py # Text chunking tests
│   └── test_roundtrip.py # End-to-end tests
├── data/                # Generated indices and metadata
│   └── redaction_report.json # Privacy filtering reports
└── README.md            # Project documentation
```

## Privacy Features

- **Document Anonymization**: Remove or mask sensitive information
- **LLM Privacy Filter**: Optional AI-based content screening
- **Local Processing**: All operations run locally (no cloud APIs)
- **Configurable Filtering**: Adjust privacy levels per use case

## Development Notes

### Core Functions

| Function | Purpose |
|----------|---------|
| `process_document_to_faiss()` | Complete ingestion pipeline |
| `retrieve_top_k_chunks()` | Semantic search and ranking |
| `load_document()` | Multi-format document parsing |
| `create_embeddings()` | Vector generation from text |
| `anonymize_text()` | Privacy filtering and anonymization |

### Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `chunk_size` | `1000` | Characters per text chunk |
| `chunk_overlap` | `200` | Overlap between chunks |
| `embedding_model` | `mxbai-embed-large` | Ollama embedding model |
| `use_llm_privacy_filter` | `false` | Enable AI-based anonymization |

## Getting Started

1. **Setup Environment**: Install UV and Ollama
2. **Install Dependencies**: Run `uv sync`
3. **Configure Models**: Pull required Ollama models
4. **Create Config**: Set up `config.json` with your preferences
5. **Ingest Documents**: Add documents to the FAISS index
6. **Query System**: Search and retrieve relevant information

## Troubleshooting

- **FAISS Issues**: Ensure `faiss-cpu` is properly installed
- **Ollama Connection**: Verify Ollama server is running (`ollama serve`)
- **Model Errors**: Check that required models are pulled locally
- **Memory Issues**: Adjust chunk sizes for large documents

## Roadmap

- **Basic RAG Pipeline** - Complete
- **FAISS Integration** - Complete
- **Privacy Filtering** - Complete
- **Batch Processing** - Planned
- **Web Interface** - Planned
- **Advanced Anonymization** - Planned
