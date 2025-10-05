#!/usr/bin/env python3
"""
CLI entry point for the privacy-preserving RAG pipeline.

Why:
    Provides a simple CLI around the anonymiser and query components.
How:
    - Loads config (JSON or YAML)
    - Commands:
        * redact <file>        → preview anonymisation or write .redacted.txt
        * ingest <file>        → anonymise + embed + persist to FAISS/metadata
        * query "<question>"   → retrieve top_k chunks from FAISS
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import faulthandler
from pathlib import Path
from typing import Any, Dict

# Fail fast, no lazy imports
import yaml  # type: ignore

from anonymiser import process_text_to_faiss, filter_all
from document_loader import read_document_to_text, detect_language
from query import query_index

# ----------------------------- Defaults -----------------------------

DEFAULT_PATHS: Dict[str, Any] = {
    "index_path": "data/index.faiss",
    "metadata_path": "data/metadata.jsonl",
    "redacted_path": "data",
    "uploads_path": "data/uploads",
}
DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "embedding_model": "mxbai-embed-large",
    "llm_pii_model_name": "mistral",
    "chunk_size": 800,
    "chunk_overlap": 120,
    "top_k": 5,
}
DEFAULT_CONFIG: Dict[str, Any] = {
    **DEFAULT_PATHS,
    **DEFAULT_LLM_CONFIG,
}

# ---------------- Logging ----------------
faulthandler.enable()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

for noisy in ("httpx", "httpcore", "uvicorn"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ------------------------ Defaults ------------------------

DEFAULT_PATHS: Dict[str, Any] = {
    "index_path": "data/index.faiss",
    "metadata_path": "data/metadata.jsonl",
    "redacted_path": "data",
    "uploads_path": "data/uploads",
}

DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "embedding_model": "mxbai-embed-large",
    "llm_pii_model_name": "mistral",
    "chunk_size": 800,
    "chunk_overlap": 120,
    "top_k": 5,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    **DEFAULT_PATHS,
    **DEFAULT_LLM_CONFIG,
}

# ----------------------------- Config -----------------------------

def load_config(path: Path | None) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML, merged over built-in defaults.
    """
    cfg = dict(DEFAULT_CONFIG)
    if path is None:
        return cfg
    if not path.exists():
        logger.warning("Config file not found at %s; using defaults.", path)
        return cfg

    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a top-level object")
        cfg.update(loaded)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        raise
    return cfg


# ----------------------------- Commands -----------------------------

def run_redact(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    """
    Redact a document and print or write to .redacted.txt.
    """
    input_path = Path(args.input_path)
    text = read_document_to_text(str(input_path))
    language = detect_language(text)

    redacted = filter_all(
        original_text=text,
        language=language,
        llm_pii_model_name=str(cfg["llm_pii_model_name"]),
        debug_report_path=None,
    )

    if args.write_file:
        out_dir = Path(cfg["redacted_path"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{input_path.stem}.redacted.txt"
        out_file.write_text(redacted, encoding="utf-8")
        print(str(out_file))
    else:
        print(redacted)
    return 0


def run_ingest(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    """
    Anonymise and ingest a document: store embeddings + metadata for retrieval.
    """
    input_path = Path(args.input_path)
    text = read_document_to_text(str(input_path))
    language = detect_language(text)

    n_new = process_text_to_faiss(
        original_text=text,
        language=language,
        source_path=input_path,
        index_path=Path(cfg["index_path"]),
        metadata_path=Path(cfg["metadata_path"]),
        redacted_path=Path(cfg["redacted_path"]),
        chunk_size=int(cfg["chunk_size"]),
        chunk_overlap=int(cfg["chunk_overlap"]),
        embedding_model=str(cfg["embedding_model"]),
        llm_pii_model_name=str(cfg["llm_pii_model_name"]),
    )

    print(f"Ingested {n_new} chunks from {input_path}")
    return 0


def run_query(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    """
    Query the FAISS index and print top_k anonymised chunks as JSON.
    """
    hits = query_index(
        question=args.question,
        index_path=Path(cfg["index_path"]),
        metadata_path=Path(cfg["metadata_path"]),
        embedding_model=str(cfg["embedding_model"]),
        top_k=int(cfg["top_k"]),
    )

    for h in hits:
        print(json.dumps(h, ensure_ascii=False))
    return 0


# ----------------------------- CLI -----------------------------

def build_parser() -> argparse.ArgumentParser:
    """Define CLI structure."""
    parser = argparse.ArgumentParser(prog="main.py", description="Privacy-preserving RAG CLI")
    parser.add_argument("--config", type=str, help="Path to config.json or .yaml", default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # redact
    pr = sub.add_parser("redact", help="Preview anonymisation of a document")
    pr.add_argument("input_path", type=str, help="Path to input file (.pdf/.docx/.txt)")
    pr.add_argument("--write-file", action="store_true", help="Write .redacted.txt instead of printing")
    pr.set_defaults(func=run_redact)

    # ingest
    pi = sub.add_parser("ingest", help="Anonymise + embed + persist to FAISS/metadata")
    pi.add_argument("input_path", type=str, help="Path to input file (.pdf/.docx/.txt)")
    pi.set_defaults(func=run_ingest)

    # query
    pq = sub.add_parser("query", help="Query anonymised FAISS index")
    pq.add_argument("question", type=str, help="Natural-language query")
    pq.set_defaults(func=run_query)

    return parser


# ----------------------------- Entry -----------------------------

def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    cfg_path = Path(args.config) if args.config else None
    cfg = load_config(cfg_path)

    try:
        return args.func(args, cfg)
    except Exception as exc:
        # Always print full traceback in case of error and exit non-zero
        import traceback
        traceback.print_exc()
        logger.error("Fatal error: %s", exc)
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        raise
