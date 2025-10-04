#!/usr/bin/env python3
# main.py
# CLI for privacy-preserving redact / ingest / query with shared config.json.
# Why: thin orchestrator that reads config, validates inputs, and calls core logic.
# How:
#   - redact <path> [--write-file] -> preview OR write /data/<stem>.redacted.txt
#   - ingest <path>                -> anonymise, embed, persist FAISS + metadata, write sidecar
#   - query "question"             -> search FAISS + metadata and return safe chunks

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# No lazy imports; fail fast
import faiss  # type: ignore
import numpy as np
import ollama

# Local modules
from anonymiser import process_text_to_faiss, filter_all
from document_loader import read_document_to_text, detect_language


# ---------------------------- Defaults ----------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "embedding_model": "mxbai-embed-large",
    "chunk_size": 800,
    "chunk_overlap": 120,
    "llm_pii_model_name": "mistral",
    # All runtime artifacts go to /data (git-ignored)
    "index_path": "data/index.faiss",
    "metadata_path": "data/metadata.jsonl",
    "redacted_path": "data",
}

ALLOWED_INPUT_SUFFIXES = {".txt", ".docx", ".pdf"}


# ---------------------------- Config / Logging ----------------------------

def load_config_file(config_path: Path) -> Dict[str, Any]:
    """
    Load config.json and overlay onto DEFAULT_CONFIG.
    Why: single source of truth for models, chunking, and paths shared by CLI and server.
    How: read JSON object if present; merge; otherwise return defaults.
    """
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
    """
    Initialise logging; controlled by CLI flag only.
    Why: predictable verbosity regardless of config content.
    How: DEBUG when --debug; otherwise INFO; concise formatter.
    """
    level = logging.DEBUG if debug_enabled else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------- Validation / Path helpers ----------------------------

def validate_input_path(path_str: str) -> Path:
    """
    Ensure input file exists and has supported extension.
    Why: fail fast with clear messages and non-zero exit on error.
    How: check existence and suffix whitelist.
    """
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix.lower() not in ALLOWED_INPUT_SUFFIXES:
        raise ValueError(f"Unsupported input type '{p.suffix}'. Allowed: {sorted(ALLOWED_INPUT_SUFFIXES)}")
    return p


def resolve_project_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """
    Resolve durable paths from config to absolute Path objects.
    Why: avoid ambiguity and ensure parent directories can be created.
    How: cast strings to Paths; return a dict of resolved paths.
    """
    return {
        "index_path": Path(str(cfg["index_path"])),
        "metadata_path": Path(str(cfg["metadata_path"])),
        "redacted_path": Path(str(cfg["redacted_path"])),  # treated as a folder
    }


# ---------------------------- Redact / Ingest core ----------------------------

def run_redact(
    input_path: Path,
    llm_model: str,
    language_hint: Optional[str],
    redacted_base: Optional[Path] = None,
) -> Tuple[str, str, Optional[Path]]:
    """
    Redact a document and optionally write a sidecar file.
    Why: preview-only by default; write sidecar when requested.
    How: read → detect language → filter_all; if redacted_base is set, save <stem>.redacted.txt.
    Returns: (filtered_text, detected_language, written_path or None)
    """
    text = read_document_to_text(input_path)
    detected_lang = language_hint or detect_language(text)
    filtered = filter_all(
        original_text=text,
        primary_lang=detected_lang,
        llm_pii_model_name=llm_model,
    )
    written_path: Optional[Path] = None
    if redacted_base is not None:
        redacted_base.mkdir(parents=True, exist_ok=True)
        written_path = redacted_base / f"{input_path.stem}.redacted.txt"
        written_path.write_text(filtered, encoding="utf-8")
    return filtered, detected_lang, written_path


def run_ingest(
    input_path: Path,
    cfg: Dict[str, Any],
    overrides: Dict[str, Any],
    language_hint: Optional[str],
) -> Dict[str, Any]:
    """
    Ingest a document: anonymise → chunk → embed → persist; also write a sidecar.
    Why: build a privacy-preserving vector store while keeping a full redacted copy for sharing.
    How: read, detect language, call process_text_to_faiss with config+overrides, return a summary dict.
    Side effects: writes FAISS index, metadata JSONL, and /data/<stem>.redacted.txt.
    """
    paths = resolve_project_paths(cfg)
    paths["redacted_path"].mkdir(parents=True, exist_ok=True)
    paths["metadata_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["index_path"].parent.mkdir(parents=True, exist_ok=True)

    text = read_document_to_text(input_path)
    primary_lang = language_hint or detect_language(text)

    chunk_size = overrides.get("chunk_size", cfg["chunk_size"])
    chunk_overlap = overrides.get("chunk_overlap", cfg["chunk_overlap"])
    embedding_model = overrides.get("embedding_model", cfg["embedding_model"])
    llm_model = overrides.get("llm_pii_model_name", cfg["llm_pii_model_name"])

    added = process_text_to_faiss(
        original_text=text,
        primary_lang=primary_lang,
        source_path=input_path,
        index_path=paths["index_path"],
        metadata_path=paths["metadata_path"],
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        embedding_model=str(embedding_model),
        llm_pii_model_name=str(llm_model),
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


# ---------------------------- Query core ----------------------------

def read_metadata_lines(meta_path: Path) -> List[Dict[str, Any]]:
    """
    Read metadata.jsonl into a list.
    Why: map FAISS vector ids back to chunk text and provenance.
    How: parse one JSON object per line; skip malformed lines.
    """
    items: List[Dict[str, Any]] = []
    if not meta_path.exists():
        return items
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                continue
    return items


def l2_normalise_np(x: np.ndarray) -> np.ndarray:
    """
    L2-normalise a matrix row-wise.
    Why: ensure dot product corresponds to cosine similarity on unit vectors.
    How: divide by Euclidean norm; protect against division by zero.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def run_query(
    question: str,
    cfg: Dict[str, Any],
    top_k: int,
    embedding_model_override: Optional[str],
) -> Dict[str, Any]:
    """
    Search FAISS for the given question and return safe chunks.
    Why: provide redacted context to public LLMs without exposing raw documents.
    How: embed question with Ollama, normalise, FAISS L2 search, map ids to metadata JSONL.
    """
    index_path = Path(str(cfg["index_path"]))
    metadata_path = Path(str(cfg["metadata_path"]))
    embedding_model = embedding_model_override or cfg["embedding_model"]

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}. Ingest documents first.")

    index = faiss.read_index(str(index_path))

    emb = ollama.embeddings(model=str(embedding_model), prompt=question)["embedding"]
    q = np.array([emb], dtype="float32")
    q = l2_normalise_np(q)

    distances, ids = index.search(q, int(top_k))
    dists = distances[0].tolist()
    vec_ids = ids[0].tolist()

    meta = read_metadata_lines(metadata_path)
    results: List[Dict[str, Any]] = []

    for dist, vid in zip(dists, vec_ids):
        if vid < 0:
            continue
        # For IndexFlatL2, distances are squared L2; on unit vectors, cos ≈ 1 - (L2^2)/2
        score = float(1.0 - (dist / 2.0))

        # Fast path: metadata aligned (vector_id == line index)
        rec: Optional[Dict[str, Any]] = meta[vid] if 0 <= vid < len(meta) else None
        if not rec or rec.get("vector_id") != vid:
            # Slow path: scan to find matching vector_id
            rec = None
            for r in meta:
                if r.get("vector_id") == vid:
                    rec = r
                    break
        if not rec:
            continue

        results.append({
            "vector_id": vid,
            "score": score,
            "chunk_text": rec.get("chunk_text", ""),
            "source_path": rec.get("source_path", ""),
            "chunk_index": rec.get("chunk_index", -1),
        })

    return {"status": "ok", "top_k": int(top_k), "results": results}


# ---------------------------- CLI ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI with subcommands and explicit flags.
    Why: thin main() and clear separation of CLI vs core logic.
    How: three subcommands: redact, ingest, query.
    """
    parser = argparse.ArgumentParser(
        description="Privacy-preserving anonymisation CLI.",
        prog="python main.py",
    )
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file (default: config.json)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subs = parser.add_subparsers(dest="command", required=True)

    # redact
    redact_p = subs.add_parser("redact", help="Preview OR write redacted sidecar")
    redact_p.add_argument("input_path", type=str, help="Path to the document (.txt/.docx/.pdf)")
    redact_p.add_argument("--language", type=str, default=None, help="Force language (nl|fr|en)")
    redact_p.add_argument("--llm-pii-model-name", type=str, default=None, help="Override config llm_pii_model_name")
    redact_p.add_argument("--write-file", action="store_true", help="Write /data/<stem>.redacted.txt instead of preview")
    redact_p.add_argument("--redacted-path", type=str, default=None, help="Override redacted_path")

    # ingest
    ingest_p = subs.add_parser("ingest", help="Anonymise, embed, persist FAISS+metadata, and write sidecar")
    ingest_p.add_argument("input_path", type=str, help="Path to the document (.txt/.docx/.pdf)")
    ingest_p.add_argument("--language", type=str, default=None, help="Force language (nl|fr|en)")
    ingest_p.add_argument("--chunk-size", type=int, default=None, help="Override chunk_size")
    ingest_p.add_argument("--chunk-overlap", type=int, default=None, help="Override chunk_overlap")
    ingest_p.add_argument("--embedding-model", type=str, default=None, help="Override embedding_model")
    ingest_p.add_argument("--llm-pii-model-name", type=str, default=None, help="Override llm_pii_model_name")
    ingest_p.add_argument("--redacted-path", type=str, default=None, help="Override redacted_path")
    ingest_p.add_argument("--index-path", type=str, default=None, help="Override index_path")
    ingest_p.add_argument("--metadata-path", type=str, default=None, help="Override metadata_path")

    # query
    query_p = subs.add_parser("query", help="Search FAISS and return safe redacted chunks")
    query_p.add_argument("question", type=str, help="Natural-language question to search for")
    query_p.add_argument("--top-k", type=int, default=5, help="Number of chunks to return (default: 5)")
    query_p.add_argument("--embedding-model", type=str, default=None, help="Override embedding_model for this query")
    query_p.add_argument("--index-path", type=str, default=None, help="Override index_path")
    query_p.add_argument("--metadata-path", type=str, default=None, help="Override metadata_path")

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
            model = args.llm_pii_model_name or cfg["llm_pii_model_name"]

            # If writing, choose output base; else preview only
            red_base: Optional[Path] = None
            if args.write_file:
                red_base = Path(args.redacted_path) if args.redacted_path else Path(str(cfg["redacted_path"]))

            filtered, lang, written_path = run_redact(
                input_path=in_path,
                llm_model=str(model),
                language_hint=args.language,
                redacted_base=red_base,
            )

            if args.write_file:
                print(f"Wrote: {written_path} (language={lang})")
            else:
                print(filtered)
            return

        if args.command == "ingest":
            in_path = validate_input_path(str(args.input_path))

            # Apply path overrides if provided
            if args.index_path:
                cfg["index_path"] = args.index_path
            if args.metadata_path:
                cfg["metadata_path"] = args.metadata_path
            if args.redacted_path:
                cfg["redacted_path"] = args.redacted_path

            overrides: Dict[str, Any] = {}
            if args.chunk_size is not None:
                overrides["chunk_size"] = args.chunk_size
            if args.chunk_overlap is not None:
                overrides["chunk_overlap"] = args.chunk_overlap
            if args.embedding_model is not None:
                overrides["embedding_model"] = args.embedding_model
            if args.llm_pii_model_name is not None:
                overrides["llm_pii_model_name"] = args.llm_pii_model_name

            summary = run_ingest(in_path, cfg, overrides, args.language)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return

        if args.command == "query":
            # Allow per-call path overrides
            if args.index_path:
                cfg["index_path"] = args.index_path
            if args.metadata_path:
                cfg["metadata_path"] = args.metadata_path

            result = run_query(
                question=str(args.question),
                cfg=cfg,
                top_k=int(args.top_k),
                embedding_model_override=args.embedding_model,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        log.error("Unknown command")
        sys.exit(1)

    except Exception as exc:
        log.error(f"{type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    
