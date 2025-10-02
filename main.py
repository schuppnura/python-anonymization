#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from pipeline import process_document_to_faiss
from query import retrieve_top_k_chunks

def main():
    parser = argparse.ArgumentParser(description="Anonymised RAG ingestion and retrieval with FAISS.")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Path to config JSON")
    subparsers = parser.add_subparsers(dest="command")

    ing = subparsers.add_parser("ingest", help="Ingest a document into FAISS")
    ing.add_argument("input_path", type=Path, help="Path to input document")

    qry = subparsers.add_parser("query", help="Query the FAISS index")
    qry.add_argument("text", type=str, help="Query text")
    qry.add_argument("--k", type=int, default=5, help="Number of results")

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
                input_path=args.input_path,
                index_path=index_path,
                metadata_path=metadata_path,
                chunk_size=int(cfg["chunk_size"]),
                chunk_overlap=int(cfg["chunk_overlap"]),
                embedding_model=str(cfg["embedding_model"]),
                use_llm_privacy_filter=bool(cfg["use_llm_privacy_filter"])
            )
            print(f"Ingested {count} chunks from {args.input_path}")
        except Exception as e:
            print(f"Ingestion error: {e}", file=sys.stderr)
            sys.exit(3)

    if args.command == "query":
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
