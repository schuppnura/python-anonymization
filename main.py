#!/usr/bin/env python3
# main.py
# Simple CLI for redact / ingest / query that relies entirely on config.json for
# models and paths. No CLI overrides of config values (except language hint and top-k).

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import faulthandler

# External libs (import early; fail fast)
import faiss  # noqa: F401
import numpy as np  # noqa: F401

# Local modules
from anonymiser import process_text_to_faiss, filter_all
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
    "redacted_path": "data",
}

DEFAULT_CONFIG: Dict[str, Any] = {**DEFAULT_LLM_CONFIG, **DEFAULT_PATHS}

ALLOWED_INPUT_SUFFIXES = {".txt", ".docx", ".pdf"}

# Show crashes even if logging is misconfigured
faulthandler.enable()  # or set env PYTHONFAULTHANDLER=1

# Reasonable default formatter; keep it simple
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


# ---------------------------- Config / Logging ----------------------------

def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load config.json and overlay onto defaults. No per-run overrides."""
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config.json must be a JSON object")
        return {**DEFAULT_CONFIG, **raw}
    except Exception as exc:
        raise RuntimeError(f"Failed to load config '{config_path}': {exc}") from exc


def set_up_logging(debug_enabled: bool) -> None:
    """Initialise logging; controlled by CLI flag only."""
    level = logging.DEBUG if debug_enabled else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------- Validation / Paths ----------------------------

def validate_input_path(path_str: str) -> Path:
    """Ensure input file exists and has supported extension."""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix.lower() not in ALLOWED_INPUT_SUFFIXES:
        raise ValueError(f"Unsupported input type '{p.suffix}'. Allowed: {sorted(ALLOWED_INPUT_SUFFIXES)}")
    return p


def resolve_project_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve durable paths from config to Path objects."""
    return {
        "index_path": Path(str(cfg["index_path"])),
        "metadata_path": Path(str(cfg["metadata_path"])),
        "redacted_path": Path(str(cfg["redacted_path"])),
    }


# ---------------------------- Redact / Ingest core ----------------------------

def run_redact(
    input_path: Path,
    cfg: Dict[str, Any],
    language_hint: Optional[str],
    write_file: bool,
) -> Tuple[str, str, Optional[Path]]:
    """
    Redact a document and optionally write a sidecar file into cfg['redacted_path'].
    Returns: (filtered_text, detected_language, written_path or None)
    """
    text = read_document_to_text(input_path)
    detected_lang = language_hint or detect_language(text)
    filtered = filter_all(
        original_text=text,
        primary_lang=detected_lang,
        llm_pii_model_name=str(cfg["llm_pii_model_name"]),
    )

    written_path: Optional[Path] = None
    if write_file:
        redacted_base = Path(str(cfg["redacted_path"]))
        redacted_base.mkdir(parents=True, exist_ok=True)
        written_path = redacted_base / f"{input_path.stem}.redacted.txt"
        written_path.write_text(filtered, encoding="utf-8")

    return filtered, detected_lang, written_path


def run_ingest(
    input_path: Path,
    cfg: Dict[str, Any],
    language_hint: Optional[str],
) -> Dict[str, Any]:
    """
    Ingest a document: anonymise → chunk → embed → persist; also write a sidecar.
    Uses only values from cfg (no CLI overrides).
    """
    paths = resolve_project_paths(cfg)
    paths["redacted_path"].mkdir(parents=True, exist_ok=True)
    paths["metadata_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["index_path"].parent.mkdir(parents=True, exist_ok=True)

    text = read_document_to_text(input_path)
    primary_lang = language_hint or detect_language(text)

    added = process_text_to_faiss(
        original_text=text,
        language=primary_lang,
        source_path=input_path,
        index_path=paths["index_path"],
        metadata_path=paths["metadata_path"],
        chunk_size=int(cfg["chunk_size"]),
        chunk_overlap=int(cfg["chunk_overlap"]),
        embedding_model=str(cfg["embedding_model"]),
        llm_pii_model_name=str(cfg["llm_pii_model_name"]),
        redacted_path=paths["redacted_path"],
    )
    return {
        "status": "ok",
        "language": primary_lang,
        "ingested_chunks": added,
        "index_path": str(paths["index_path"]),
        "metadata_path": str(paths["metadata_path"]),
        "redacted_path": str(paths["redacted_path"]),
    }


# ---------------------------- CLI ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI without config overrides."""
    parser = argparse.ArgumentParser(
        description="Privacy-preserving anonymisation CLI (no config overrides).",
        prog="python main.py",
    )
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file (default: config.json)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subs = parser.add_subparsers(dest="command", required=True)

    # redact
    redact_p = subs.add_parser("redact", help="Preview OR write redacted sidecar")
    redact_p.add_argument("input_path", type=str, help="Path to the document (.txt/.docx/.pdf)")
    redact_p.add_argument("--language", type=str, default=None, help="Optional language hint (nl|fr|en)")
    redact_p.add_argument("--write-file", action="store_true", help="Write /data/<stem>.redacted.txt instead of preview")

    # ingest
    ingest_p = subs.add_parser("ingest", help="Anonymise, embed, persist FAISS+metadata, and write sidecar")
    ingest_p.add_argument("input_path", type=str, help="Path to the document (.txt/.docx/.pdf)")
    ingest_p.add_argument("--language", type=str, default=None, help="Optional language hint (nl|fr|en)")

    # query
    query_p = subs.add_parser("query", help="Search FAISS and return safe redacted chunks")
    query_p.add_argument("question", type=str, help="Natural-language question to search for")
    query_p.add_argument("--top-k", type=int, default=5, help="Number of chunks to return (default: 5)")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    set_up_logging(debug_enabled=args.debug)
    log = logging.getLogger("main")

    try:
        cfg = load_config_file(Path(str(args.config)))
    except Exception as exc:
        log.error(str(exc))
        sys.exit(1)

    try:
        if args.command == "redact":
            in_path = validate_input_path(str(args.input_path))
            filtered, lang, written_path = run_redact(
                input_path=in_path,
                cfg=cfg,
                language_hint=args.language,
                write_file=bool(args.write_file),
            )
            if args.write_file:
                print(f"Wrote: {written_path} (language={lang})")
            else:
                print(filtered)
            return

        if args.command == "ingest":
            in_path = validate_input_path(str(args.input_path))
            summary = run_ingest(
                input_path=in_path,
                cfg=cfg,
                language_hint=args.language,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return

        if args.command == "query":
            result = search_index_and_collect_results(
                question=str(args.question),
                index_path=Path(str(cfg["index_path"])),
                metadata_path=Path(str(cfg["metadata_path"])),
                embedding_model=str(cfg["embedding_model"]),
                top_k=int(args.top_k),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        log.error("Unknown command")
        sys.exit(1)

    except Exception as exc:
        log.error(f"{type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Always print a full traceback to stderr
        import traceback
        traceback.print_exc()
        # And make sure process exits non-zero
        sys.exit(1)
