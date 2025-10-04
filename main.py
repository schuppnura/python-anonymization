# main.py
# Orchestration script for redact / ingest / query using anonymiser and document loader.

import argparse
import json
import logging
import sys
import traceback
import faulthandler
from pathlib import Path

from anonymiser import filter_all, process_text_to_faiss
from document_loader import read_document_to_text, detect_language
from query import query_index  # implemented separately

faulthandler.enable()

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

for noisy in ("httpx", "httpcore", "uvicorn"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------- Default Config ----------------
DEFAULT_LLM_CONFIG = {
    "llm_pii_model_name": "mistral",
    "embedding_model": "mxbai-embed-large",
}

DEFAULT_PATHS = {
    "index_path": Path("data/index.faiss"),
    "metadata_path": Path("data/metadata.jsonl"),
    "redacted_path": Path("data"),   # redacted text output directory
    "chunk_size": 800,
    "chunk_overlap": 120,
}

# ---------------- Orchestration ----------------

def run_redact(args, cfg):
    """
    Redact a document and either print the filtered text or write it to disk.

    Why:
        The redact mode is for QA and quick verification of anonymisation
        without indexing or persistence.
    How:
        - Load document text
        - Detect its primary language
        - Run anonymisation filter_all()
        - Print to stdout or write to a .redacted.txt file under /data
    Side effects:
        Can create a redacted sidecar file on disk.
    """
    try:
        text = read_document_to_text(args.input_path)
        language = detect_language(text)
        logger.info("Detected language: %s", language)

        filtered = filter_all(
            original_text=text,
            language=language,
            llm_pii_model_name=cfg["llm_pii_model_name"],
            debug_report_path=Path("data") / "redaction_report.json",
        )

        if args.write_file:
            out_name = f"{Path(args.input_path).stem}.redacted.txt"
            out_path = DEFAULT_PATHS["redacted_path"] / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(filtered, encoding="utf-8")
            logger.info("Redacted file written to %s", out_path)
        else:
            print(filtered)

    except Exception:
        logger.exception("Redact error")
        raise


def run_ingest(args, cfg):
    """
    Ingest a document into the FAISS vector store with anonymisation applied.

    Why:
        Ingestion is the RAG pipeline: it anonymises text, splits into chunks,
        computes embeddings, and persists them to FAISS + metadata JSONL.
    How:
        - Read original text and detect language
        - Call process_text_to_faiss with paths, models, chunking config
        - Write updated FAISS index and metadata
    Side effects:
        Updates FAISS index and metadata.jsonl under /data.
    """
    try:
        text = read_document_to_text(args.input_path)
        language = detect_language(text)
        logger.info("Detected language: %s", language)

        n_new = process_text_to_faiss(
            original_text=text,
            language=language,
            source_path=Path(args.input_path),
            index_path=DEFAULT_PATHS["index_path"],
            metadata_path=DEFAULT_PATHS["metadata_path"],
            redacted_path=DEFAULT_PATHS["redacted_path"],
            chunk_size=DEFAULT_PATHS["chunk_size"],
            chunk_overlap=DEFAULT_PATHS["chunk_overlap"],
            embedding_model=cfg["embedding_model"],
            llm_pii_model_name=cfg["llm_pii_model_name"],
        )
        logger.info("Ingested %d new chunks from %s", n_new, args.input_path)

    except Exception:
        logger.exception("Ingestion error")
        raise


def run_query(args, cfg):
    """
    Query the FAISS index with a natural-language query.

    Why:
        Query mode tests the RAG setup and retrieves top-k chunks
        that were anonymised and embedded during ingestion.
    How:
        - Embed the query with the same embedding model
        - Perform FAISS search
        - Print JSON results with chunk_text and provenance
    Side effects:
        None (read-only).
    """
    try:
        results = query_index(
            query_text=args.query,
            index_path=DEFAULT_PATHS["index_path"],
            metadata_path=DEFAULT_PATHS["metadata_path"],
            embedding_model=cfg["embedding_model"],
            top_k=args.top_k,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))

    except Exception:
        logger.exception("Query error")
        raise

# ---------------- Main ----------------

def main():
    """
    Command-line entry point for the anonymisation pipeline.

    Supports three subcommands:
    - redact: Preview redaction only
    - ingest: Anonymise and store vectors
    - query: Query the FAISS store for similar chunks
    """
    parser = argparse.ArgumentParser(description="Anonymiser pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # redact
    redact_parser = subparsers.add_parser("redact", help="Redact PII from a document")
    redact_parser.add_argument("input_path", type=str, help="Path to document")
    redact_parser.add_argument("--write-file", action="store_true", help="Write redacted text to /data")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document into FAISS")
    ingest_parser.add_argument("input_path", type=str, help="Path to document")

    # query
    query_parser = subparsers.add_parser("query", help="Query the FAISS index")
    query_parser.add_argument("query", type=str, help="Query text")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()
    cfg = {**DEFAULT_LLM_CONFIG}

    if args.command == "redact":
        run_redact(args, cfg)
    elif args.command == "ingest":
        run_ingest(args, cfg)
    elif args.command == "query":
        run_query(args, cfg)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        raise
