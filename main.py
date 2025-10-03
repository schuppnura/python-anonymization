#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from constants import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH
from anonymiser import apply_regex_redaction, apply_person_redaction_with_llm
from pipeline import process_document_to_faiss
from query import retrieve_top_k_chunks
from document_loader import read_document_to_text

def main():
    parser = argparse.ArgumentParser(description="Anonymised RAG ingestion and retrieval with FAISS.")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Path to config JSON")
    subparsers = parser.add_subparsers(dest="command")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document into FAISS index")
    ingest_parser.add_argument("input_path", type=str, help="Path to document")

    # query
    query_parser = subparsers.add_parser("query", help="Query the FAISS index")
    query_parser.add_argument("query_text", type=str, help="Query string")
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        with args.config.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(2)

    index_path = Path(cfg["index_path"])
    metadata_path = Path(cfg["metadata_path"])

    if args.command == "ingest":
        try:
            count = process_document_to_faiss(
                input_path=Path(args.input_path),
                index_path=Path(cfg.get("index_path", DEFAULT_INDEX_PATH)),
                metadata_path=Path(cfg.get("metadata_path", DEFAULT_METADATA_PATH)),
                chunk_size=int(cfg["chunk_size"]),
                chunk_overlap=int(cfg["chunk_overlap"]),
                embedding_model=str(cfg["embedding_model"]),
                llm_person_redaction_enabled=bool(cfg.get("llm_person_redaction_enabled", False)),
                llm_person_model_name=str(cfg.get("llm_person_model_name", "mistral"))
            )
            print(f"Ingested {count} chunks from {args.input_path}")
        except Exception as e:
            print(f"Ingestion error: {e}", file=sys.stderr)
            sys.exit(3)

    elif args.command == "query":
        try:
            hits = retrieve_top_k_chunks(
                query_text=args.text,
                k=args.k,
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model=str(cfg["embedding_model"])
            )
            for h in hits:
                print(f"[{h['rank']}] score={h['score']:.4f} path={h['source_path']} chunk={h['chunk_index']}")
                print(h["chunk_text"])
                print("-" * 60)
        except Exception as e:
            print(f"Query error: {e}", file=sys.stderr)
            sys.exit(4)
  
if __name__ == "__main__":
    main()
