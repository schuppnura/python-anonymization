# mcp_server.py
# FastAPI MCP-style server for redact / ingest / query with privacy-preserving RAG.
# Why: expose a local, anonymising interface that a public LLM client can call safely.
# How: load a single config.json at startup; do not accept per-request overrides for models/paths.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# External libs (import early; fail fast)
import faiss  # noqa: F401
import numpy as np  # noqa: F401
import ollama  # noqa: F401

# Local modules (no lazy imports)
from anonymiser import filter_all, process_text_to_faiss
from document_loader import read_document_to_text, detect_language
from query import search_index_and_collect_results


# ---------------------------- Default Config ----------------------------

DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "embedding_model": "mxbai-embed-large",
    "chunk_size": 800,
    "chunk_overlap": 120,
    "llm_pii_model_name": "mistral",
}

DEFAULT_PATHS: Dict[str, Any] = {
    # All runtime artifacts live under /data (git-ignored)
    "index_path": "data/index.faiss",
    "metadata_path": "data/metadata.jsonl",
    "redacted_path": "data",  # redacted sidecars go straight into /data
}

DEFAULT_CONFIG: Dict[str, Any] = {**DEFAULT_LLM_CONFIG, **DEFAULT_PATHS}


# ---------------------------- App and Config Load ----------------------------

app = FastAPI(title="Anonymiser MCP", version="2.0.0")
CONFIG_PATH = Path("config.json")

def load_server_config() -> Dict[str, Any]:
    """
    Load the server configuration once at startup.

    Why: a single source of truth for models, chunking, and persistence paths;
         consistent with the CLI design (no per-request overrides).
    How: read config.json if present; merge onto defaults; fall back to defaults on error.
    """
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config.json must be a JSON object")
        return {**DEFAULT_CONFIG, **raw}
    except Exception as exc:
        print(f"Warning: failed to load {CONFIG_PATH}: {exc}")
        return DEFAULT_CONFIG.copy()

SERVER_CONFIG = load_server_config()

INDEX_PATH = Path(str(SERVER_CONFIG["index_path"]))
METADATA_PATH = Path(str(SERVER_CONFIG["metadata_path"]))
REDACTED_PATH = Path(str(SERVER_CONFIG["redacted_path"]))

# Temporary upload area (under /data)
UPLOAD_DIR = Path("data") / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Ensure durable parents exist
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
REDACTED_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------- Request / Response Models ----------------------------

class RedactRequest(BaseModel):
    """
    /redact request schema.

    Why: validate input and keep the interface explicit.
    How: accept text plus optional language hint; config drives models/paths.
    """
    text: str
    language_hint: Optional[str] = Field(default=None, description="Optional language hint (nl|fr|en)")


class IngestRequest(BaseModel):
    """
    /ingest request schema.

    Why: anonymise and persist in one call, using server config only.
    How: source_label is used for provenance and redacted file naming.
    """
    text: str
    source_label: str = "uploaded"
    language_hint: Optional[str] = Field(default=None, description="Optional language hint (nl|fr|en)")


class QueryRequest(BaseModel):
    """
    /query request schema.

    Why: provide safe chunks for RAG.
    How: only top_k is adjustable per request; embeddings model is from config.
    """
    question: str
    top_k: int = 5


# ---------------------------- Helper Functions ----------------------------

def write_redacted_sidecar(filtered_text: str, source_label: str) -> Path:
    """
    Write a full redacted sidecar file into REDACTED_PATH with a stable name.

    Why: provide a durable, shareable, fully redacted copy per ingested source.
    How: base on source_label stem; ensure directory exists; use UTF-8.
    Side effects: writes <redacted_path>/<stem>.redacted.txt
    """
    REDACTED_PATH.mkdir(parents=True, exist_ok=True)
    stem = Path(source_label).stem or "document"
    out_path = REDACTED_PATH / f"{stem}.redacted.txt"
    out_path.write_text(filtered_text, encoding="utf-8")
    return out_path


# ---------------------------- Endpoints ----------------------------

@app.get("/config")
def get_config():
    """
    Return the effective server configuration.

    Why: transparency and sanity checks for clients.
    How: echo merged defaults + config.json.
    """
    return SERVER_CONFIG


@app.post("/redact")
def redact(req: RedactRequest):
    """
    Redact plain text without indexing.

    Why: quick QA or one-off redactions. Clients can store or display the safe text.
    How: detect language if absent; run filter_all with the configured LLM; return filtered text.
    """
    if not req.text or not isinstance(req.text, str):
        raise HTTPException(status_code=400, detail="Field 'text' must be a non-empty string.")

    primary_lang = req.language_hint or detect_language(req.text)
    filtered = filter_all(
        original_text=req.text,
        primary_lang=primary_lang,
        llm_pii_model_name=str(SERVER_CONFIG["llm_pii_model_name"]),
    )
    return {"status": "ok", "language": primary_lang, "filtered_preview": filtered}


@app.post("/ingest")
def ingest(req: IngestRequest):
    """
    Ingest text: anonymise → chunk → embed → persist; also write a redacted sidecar.

    Why: build a privacy-preserving vector store; keep a full shareable redacted file.
    How: use only server configuration for models and paths; no per-request overrides.
    """
    if not req.text or not isinstance(req.text, str):
        raise HTTPException(status_code=400, detail="Field 'text' must be a non-empty string.")

    primary_lang = req.language_hint or detect_language(req.text)

    # Run full filter pipeline (inside process_text_to_faiss)
    try:
        added = process_text_to_faiss(
            original_text=req.text,
            primary_lang=primary_lang,
            source_path=Path(req.source_label),
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH,
            chunk_size=int(SERVER_CONFIG["chunk_size"]),
            chunk_overlap=int(SERVER_CONFIG["chunk_overlap"]),
            embedding_model=str(SERVER_CONFIG["embedding_model"]),
            llm_pii_model_name=str(SERVER_CONFIG["llm_pii_model_name"]),
            redacted_path=REDACTED_PATH,  # sidecar goes straight under /data
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return {
        "status": "ok",
        "language": primary_lang,
        "ingested_chunks": added,
        "index_path": str(INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
        "redacted_path": str(REDACTED_PATH),
        "source_label": req.source_label,
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    mode: str = Form("redact"),                 # "redact" | "ingest"
    source_label: str = Form("upload"),
    language_hint: Optional[str] = Form(None),
):
    """
    Accept a .docx or .pdf, extract text server-side, and either redact or ingest.

    Why: keep raw documents local; clients never send extracted text elsewhere.
    How: save to ./data/uploads, extract with document_loader, then call anonymiser functions.
    """
    filename = file.filename or "upload"
    suffix = filename.lower().split(".")[-1]
    if suffix not in {"docx", "pdf"}:
        raise HTTPException(status_code=400, detail="Only .docx and .pdf are supported.")

    dest = UPLOAD_DIR / filename
    try:
        content = await file.read()
        dest.write_bytes(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to save file: {exc}")

    try:
        text = read_document_to_text(dest)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read document: {exc}")

    primary_lang = language_hint or detect_language(text)

    if mode.lower() == "redact":
        filtered = filter_all(
            original_text=text,
            primary_lang=primary_lang,
            llm_pii_model_name=str(SERVER_CONFIG["llm_pii_model_name"]),
        )
        return {
            "status": "ok",
            "mode": "redact",
            "language": primary_lang,
            "source_label": source_label or filename,
            "filtered_preview": filtered,
        }

    if mode.lower() == "ingest":
        try:
            added = process_text_to_faiss(
                original_text=text,
                primary_lang=primary_lang,
                source_path=Path(source_label or filename),
                index_path=INDEX_PATH,
                metadata_path=METADATA_PATH,
                chunk_size=int(SERVER_CONFIG["chunk_size"]),
                chunk_overlap=int(SERVER_CONFIG["chunk_overlap"]),
                embedding_model=str(SERVER_CONFIG["embedding_model"]),
                llm_pii_model_name=str(SERVER_CONFIG["llm_pii_model_name"]),
                redacted_path=REDACTED_PATH,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

        return {
            "status": "ok",
            "mode": "ingest",
            "language": primary_lang,
            "source_label": source_label or filename,
            "ingested_chunks": added,
            "index_path": str(INDEX_PATH),
            "metadata_path": str(METADATA_PATH),
            "redacted_path": str(REDACTED_PATH),
        }

    raise HTTPException(status_code=400, detail="mode must be 'redact' or 'ingest'")


@app.post("/query")
def query(req: QueryRequest):
    """
    Query the FAISS index and return safe redacted chunks.

    Why: let public LLM clients retrieve privacy-preserving context for RAG.
    How: embed question with configured embedding model; search; map to metadata.jsonl.
    """
    if req.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0")

    try:
        result = search_index_and_collect_results(
            question=req.question,
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH,
            embedding_model=str(SERVER_CONFIG["embedding_model"]),
            top_k=int(req.top_k),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    return result


@app.get("/redacted/{basename}")
def get_redacted(basename: str):
    """
    Download a redacted sidecar (.txt) by name.

    Why: easy retrieval and sharing of the safe full-text version.
    How: restrict to the configured redacted_path and .txt extension.
    """
    safe_dir = REDACTED_PATH.resolve()
    candidate = (REDACTED_PATH / basename).resolve()
    if candidate.parent != safe_dir or not candidate.name.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Invalid file name")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(candidate), media_type="text/plain", filename=candidate.name)
