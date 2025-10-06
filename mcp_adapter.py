#!/usr/bin/env python3
#mcp_adapter.py
"""
MCP Adapter for the Anonymization RAG system.

This module exposes a Model Context Protocol (MCP) interface using FastMCP,
so that public LLM clients (like Claude or ChatGPT MCP-compatible agents)
can call into the local anonymization and retrieval pipeline.

It acts as a bridge layer between the FastAPI app and the internal modules
(anonymiser, query, and document ingestion).

Run locally with:
    uv run mcp_adapter.py
or
    python mcp_adapter.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP  # make sure to `uv pip install fastmcp`

# Local imports (no lazy imports)
from anonymiser import filter_all, process_text_to_faiss
from query import query_index
from document_loader import read_document_to_text

# ---------------------------------------------------------------------
# MCP APP SETUP
# ---------------------------------------------------------------------

mcp = FastMCP("privacy-rag")


# ---------------------------------------------------------------------
# TOOLS (endpoints)
# ---------------------------------------------------------------------

@mcp.tool()
def get_config() -> Dict[str, Any]:
    """
    Return the current runtime configuration.

    Why:
        External MCP clients may want to query system state (paths, model names).
    How:
        Read from config.json if present, otherwise return defaults.
    """
    cfg_path = Path("config.json")
    if not cfg_path.exists():
        return {
            "index_path": "data/index.faiss",
            "metadata_path": "data/metadata.jsonl",
            "embedding_model": "mxbai-embed-large",
            "llm_pii_model_name": "mistral",
            "redacted_path": "data",
        }

    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@mcp.tool()
def redact_text(text: str, language_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Redact sensitive information in a given text.

    Why:
        This allows MCP clients to use the anonymisation service directly
        without uploading files.
    How:
        Passes the text through the `filter_all` pipeline using the configured LLM.
    """
    cfg = get_config()
    llm_model = cfg.get("llm_pii_model_name", "mistral")
    lang = language_hint or "nl"

    redacted = filter_all(
        original_text=text,
        primary_lang=lang,
        llm_pii_model_name=llm_model,
    )

    return {"language": lang, "redacted_text": redacted}


@mcp.tool()
def ingest_file(
    path: str,
    source_label: Optional[str] = None,
    language_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingest a file (PDF/DOCX) into the FAISS index after anonymisation.

    Why:
        Enables local privacy-preserving ingestion for RAG retrieval.
    How:
        - Loads document text.
        - Redacts sensitive content.
        - Chunks and embeds with Ollama.
        - Updates FAISS index and metadata.
    """
    cfg = get_config()
    lang = language_hint or "nl"
    text = read_document_to_text(str(path))

    n_new = process_text_to_faiss(
        original_text=text,
        language=lang,
        source_path=Path(source_label or path),
        index_path=Path(cfg["index_path"]),
        metadata_path=Path(cfg["metadata_path"]),
        redacted_path=Path(cfg["redacted_path"]),
        chunk_size=int(cfg.get("chunk_size", 800)),
        chunk_overlap=int(cfg.get("chunk_overlap", 120)),
        embedding_model=str(cfg.get("embedding_model", "mxbai-embed-large")),
        llm_pii_model_name=str(cfg.get("llm_pii_model_name", "mistral")),
    )

    return {"ingested_vectors": n_new, "source": path, "language": lang}


@mcp.tool()
def query_corpus(question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Query the FAISS index for relevant document chunks.

    Why:
        Enables retrieval-augmented generation (RAG) workflows for LLM clients.
    How:
        Executes a semantic similarity search using the configured embedding model.
    """
    cfg = get_config()
    index_path = Path(cfg["index_path"])
    metadata_path = Path(cfg["metadata_path"])
    model = cfg.get("embedding_model", "mxbai-embed-large")
    k = top_k or int(cfg.get("top_k", 5))

    results = query_index(
        question=question,
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model=model,
        top_k=k,
    )

    return results


# ---------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """
    Launch the MCP server.
    This will expose the tools for discovery by compatible LLM clients.
    """
    mcp.run()
