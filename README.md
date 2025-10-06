# Anonymisation RAG System

A privacy-first document processing and retrieval system that combines local AI models (via Ollama) with FAISS vector storage to enable secure, GDPR-compliant retrieval-augmented generation (RAG).

Please refer to our blog that elaborates on the why's and how's of this project: [Feed LLMs Carefully: A Filter-First RAG Architecture](https://nura.pro/2025/10/04/feed-llms-carefully-a-filter-first-rag-architecture/)

**Version 2.0.0** — Includes an MCP-style FastAPI server, document upload & redaction endpoints, stronger privacy filtering, and unified configuration management.

## Overview

### Execution Modes

- CLI Mode (`main.py`) — Run local ingestion, redaction, and testing.
- API Mode (`mcp_server.py`) — Expose the same functionality through an MCP-compatible FastAPI server.
- MCP Mode (`mcp_adapter.py`) — Native Model Context Protocol server using FastMCP for direct LLM client integration.

All modes share the same anonymisation and FAISS back-end logic.

## How It Works

The Anonymiser acts as a trusted layer between your personal data and the public LLM. Here is how it works:

1. You use an LLM client (e.g. Claude or an LLM wrapper) and enter a prompt that needs a private document

2. The LLM client calls the anonymiser via the standard MCP interface and requests the relevant document chunks

3. The anonymiser manages the prompt locally with its own LLM (e.g. Mistral) and queries a vector store it has pre-built and locally curated

4. The anonymiser reconstructs the curated, relevant document chunks and feeds these back to the LLM client as context

5. The LLM client builds the final prompt with this context and submits it to the public LLM service, without leaking any private data

6. The LLM responds with context-aware output

## Features

- Privacy-First RAG — Only anonymised text is embedded or indexed.
- Local AI — Uses Ollama for both embeddings and PII-aware redaction.
- Multi-Format Input — Supports PDF and DOCX files.
- Sentence-Aware Chunking — Improved retrieval precision.
- High-Performance FAISS Index — Persistent vector store for semantic search.
- Modular Architecture — Easily extendable detectors and pipelines.

## Data Flow

[Upload / Ingest]
      ↓
[Anonymiser → Redacted Text]
      ↓
[Chunking → Embedding → FAISS Index + Metadata]

[Query]
      ↓
[FAISS Search → Filtered Chunks → Optional LLM Answer Layer]

Privacy Guarantee: All indexed content and retrieval results are derived exclusively from filtered text.
No raw PII is embedded, stored, or returned.

## Architecture

| Component | Role |
|------------|-------|
| Document Loader | Extracts text from PDF and DOCX files |
| Anonymiser | Removes or masks PII via regex and LLM |
| Chunking | Splits text into sentence-aligned segments |
| Embedding | Creates vectors using Ollama embedding models |
| FAISS Store | Persists vector index for retrieval |
| Query Engine | Finds top-k relevant chunks for questions |

## Setup

### Prerequisites
- Python 3.11 or newer
- UV package manager (https://docs.astral.sh/uv/getting-started/installation/)
- Ollama runtime (https://ollama.ai) running locally

### Installation

```bash
cd Anonymization
uv sync
uv sync --group dev
```

UV automatically uses the virtual environment configured in .env. No cloud dependencies are required.

## Required Models (Example)

```bash
ollama pull mxbai-embed-large
ollama pull llama3.1:8b-instruct-q4_K_M
```

## Quick Start

```bash
uv run main.py ingest data/Zorgvolmacht.docx
uv run main.py query "Wie zijn de lasthebbers?"
curl -s -X POST http://127.0.0.1:8000/redact -H "Content-Type: application/json" -d '{"text":"De heer Carlo Schupp, geboren te Brussel op 2 mei 1980.","language_hint":"nl"}' | jq -r .filtered_preview
```

## API Server (MCP Style)

Run the FastAPI server:

```bash
uvicorn mcp_server:app --host 127.0.0.1 --port 8000
```

### Endpoints

| Method | Endpoint | Purpose |
|---------|-----------|---------|
| GET | /config | Return current configuration |
| POST | /redact | Preview anonymisation of raw text |
| POST | /upload (mode=redact) | Upload file and get redacted version |
| POST | /upload (mode=ingest) | Upload and ingest file into index |
| POST | /query | Semantic search against filtered index |
| GET | /redacted/{filename} | Download redacted sidecar file |

Interactive docs: http://127.0.0.1:8000/docs
OpenAPI schema: http://127.0.0.1:8000/openapi.json

## MCP Server (FastMCP)

The MCP adapter provides a native Model Context Protocol interface using FastMCP, enabling direct integration with MCP-compatible LLM clients like Claude Desktop, Cline, or other MCP-aware applications.

### Running the MCP Server

```bash
uv run mcp_adapter.py
```

Or run in the background:

```bash
nohup uv run mcp_adapter.py > mcp_adapter.log 2>&1 &
```

### MCP Tools Available

| Tool | Purpose |
|------|----------|
| `get_config` | Retrieve current system configuration |
| `redact_text` | Anonymize sensitive information in provided text |
| `ingest_file` | Process and ingest documents into the FAISS index |
| `query_corpus` | Perform semantic search against the document corpus |

### Connecting MCP Clients

For MCP-compatible clients, configure the server connection:

```json
{
  "mcpServers": {
    "privacy-rag": {
      "command": "uv",
      "args": ["run", "mcp_adapter.py"],
      "cwd": "/path/to/Anonymization"
    }
  }
}
```

The MCP server exposes the same core functionality as the API and CLI modes but through the standardized MCP protocol for seamless LLM integration.

## Configuration

Example config.json:

```json
{
  "index_path": "data/index.faiss",
  "metadata_path": "data/metadata.jsonl",
  "chunk_size": 800,
  "chunk_overlap": 120,
  "embedding_model": "mxbai-embed-large",
  "llm_pii_model_name": "mistral",
  "redacted_path": "data"
}
```

Configuration Precedence:
1. config.json — primary source of truth
2. Runtime inputs (language_hint, debug) — temporary overrides
3. Both CLI and API modes share the same configuration loader

## CLI Commands

**Ingest a document**
```bash
uv run main.py ingest path/to/document.pdf
```

**Query the index**
```bash
uv run main.py query "What is the main topic?" --k 5
```

## Project Structure

```
Anonymization/
├── main.py
├── mcp_server.py
├── mcp_adapter.py
├── query.py
├── document_loader.py
├── anonymiser.py
├── config.json
├── tests/
├── data/
│   ├── uploads/
│   ├── metadata.jsonl
│   ├── index.faiss
│   └── *.redacted.txt
└── README.md
```

## Development Tasks

```bash
uv run black .
uv run ruff check .
uv run mypy .
uv run pytest
```

## Extending the System

- Add new regex detectors: edit detect_identifier_regex_spans()
- Add new LLM rules: extend BIRTH_ADDR_SYSTEMPROMPT templates
- Swap embedding models: update embedding_model in config.json
- Add custom language support: extend tokenize_sentences() and abbreviation sets

## Troubleshooting

| Issue | Cause / Fix |
|--------|-------------|
| FAISS not found | Ensure faiss-cpu installed (uv add faiss-cpu) |
| Ollama errors | Verify ollama serve is running |
| Missing models | Run ollama pull <model> |
| Memory overflow | Lower chunk_size in config |

## License & Attribution

Copyright © 2025 Nura BV.  
All rights reserved.  

This software and its documentation are provided under the MIT License.  
Use is permitted with attribution to Nura and the original authors.  
