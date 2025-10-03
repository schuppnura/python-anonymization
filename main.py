#!/usr/bin/env python3
# main.py
#
# Entry point for anonymisation + RAG system.
# Loads config (JSON or YAML), dispatches CLI commands, orchestrates document_loader + anonymiser.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# No lazy imports: import all dependencies at module import time.
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # We’ll fail fast if user passes a YAML config

from document_loader import read_document_to_text, detect_language
from anonymiser import (
    process_text_to_faiss,
    filter_identifier_regex,
    filter_birth_and_address_regex,
    filter_birth_and_address_llm,
    filter_person_name_llm,
)
import query


# ---------------------------- Config defaults ----------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "llm_person_model_name": "mistral",
    "llm_person_redaction_enabled": True,
    "embedding_model": "mxbai-embed-large",
    "chunk_size": 800,
    "chunk_overlap": 120,
    "index_path": "data/index.faiss",
    "metadata_path": "data/metadata.jsonl",
    "log_level": "INFO",
}

YAML_SUFFIXES = {".yaml", ".yml"}


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML and merge with defaults.
    - If file is missing, returns defaults.
    - If YAML requested and PyYAML is unavailable, exits with a clear error.
    """
    if not config_path.exists():
        return dict(DEFAULT_CONFIG)

    ext = config_path.suffix.lower()
    try:
        if ext in YAML_SUFFIXES:
            if yaml is None:
                print(
                    "YAML config requested but PyYAML is not installed. "
                    "Install with: pip install pyyaml",
                    file=sys.stderr,
                )
                sys.exit(2)
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    raise ValueError("Top-level YAML must be a mapping/object")
        else:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
                if not isinstance(data, dict):
                    raise ValueError("Top-level JSON must be an object")
    except Exception as e:
        print(f"Error loading config {config_path}: {e}", file=sys.stderr)
        sys.exit(2)

    merged = dict(DEFAULTS)
    merged.update(data or {})
    return merged


# ---------------------------- CLI commands ----------------------------

def run_ingest(args, cfg):
    try:
        src_path = Path(args.input_path)
        text = read_document_to_text(src_path)
        primary_lang, _scores = detect_language(text)

        added = process_text_to_faiss(
            original_text=text,
            primary_lang=primary_lang,
            source_path=src_path,
            index_path=Path(cfg["index_path"]),
            metadata_path=Path(cfg["metadata_path"]),
            chunk_size=int(cfg["chunk_size"]),
            chunk_overlap=int(cfg["chunk_overlap"]),
            embedding_model=str(cfg["embedding_model"]),
            llm_person_redaction_enabled=bool(cfg["llm_person_redaction_enabled"]),
            llm_person_model_name=str(cfg["llm_person_model_name"]),
        )
        print(f"Ingested {added} chunks from {args.input_path}")
    except Exception as e:
        print(f"Ingestion error: {e}", file=sys.stderr)
        sys.exit(1)


def run_query(args, cfg):
    try:
        results = query.retrieve_top_k_chunks(
            query_text=args.text,
            index_path=Path(cfg["index_path"]),
            metadata_path=Path(cfg["metadata_path"]),
            embedding_model=str(cfg["embedding_model"]),
            k=int(args.k),
        )
        # JSON per line for easy piping
        for row in results:
            print(json.dumps(row, ensure_ascii=False))
    except Exception as e:
        print(f"Query error: {e}", file=sys.stderr)
        sys.exit(1)


def run_redact(args, cfg):
    """QA command: show preview of redaction without indexing."""
    try:
        src_path = Path(args.input_path)
        text = read_document_to_text(src_path)
        primary_lang, _scores = detect_language(text)

        red = filter_identifier_regex(text)
        red = filter_birth_and_address_regex(red, [primary_lang])
        red = filter_birth_and_address_llm(
            red,
            model_name=str(cfg.get("llm_person_model_name", "mistral")),
            enabled=bool(cfg.get("llm_person_redaction_enabled", False)),
            primary_lang=primary_lang,
        )
        red = filter_person_name_llm(
            red,
            model_name=str(cfg.get("llm_person_model_name", "mistral")),
            enabled=bool(cfg.get("llm_person_redaction_enabled", False)),
        )

        print("=== Original (first 1000 chars) ===")
        print(text[:1000])
        print("\n=== Redacted (first 1000 chars) ===")
        print(red[:1000])
    except Exception as e:
        print(f"Redact error: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------- Main ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Anonymisation + RAG CLI (JSON/YAML config)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json|.yaml|.yml")
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ing = sub.add_parser("ingest", help="Ingest a document into FAISS")
    p_ing.add_argument("input_path", type=str, help="Path to document (.txt/.pdf/.docx)")

    # query
    p_q = sub.add_parser("query", help="Query FAISS index")
    p_q.add_argument("text", type=str, help="Query text")
    p_q.add_argument("-k", type=int, default=5, help="Number of results to return")

    # redact (QA)
    p_r = sub.add_parser("redact", help="Preview redaction (no indexing)")
    p_r.add_argument("input_path", type=str, help="Path to document (.txt/.pdf/.docx)")

    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    if args.command == "ingest":
        run_ingest(args, cfg)
    elif args.command == "query":
        run_query(args, cfg)
    elif args.command == "redact":
        run_redact(args, cfg)


if __name__ == "__main__":
    main()
