# query.py
# Retrieval utilities for the anonymised RAG index.
#
# Functions:
#   retrieve_top_k_chunks(query_text, index_path, metadata_path, embedding_model, k=5) -> list[dict]
#     - Returns [{ "vector_id": int, "score": float, "chunk_text": str }, ...]
#
# Design rules:
# - Functions start with verbs; no nested functions
# - Validate inputs early; clear errors
# - Separate I/O from core logic; return data, don’t print
# - Comments explain why; state assumptions and side effects

from __future__ import annotations

from pathlib import Path
import json

# FAISS is required for querying persisted vectors
try:
    import faiss  # type: ignore
except Exception as e:
    faiss = None

import numpy as np

# Reuse the exact embedding stack used for ingestion to ensure vector space compatibility
from anonymiser import create_embeddings, l2_normalise


def _load_index(index_path: Path):
    """
    Load a FAISS index from disk.
    Assumptions:
      - File exists and is a valid FAISS index.
    """
    if faiss is None:
        raise RuntimeError("faiss not installed")
    p = Path(index_path)
    if not p.exists():
        raise FileNotFoundError(f"Index not found: {p}")
    return faiss.read_index(str(p))


def _read_metadata_subset(metadata_path: Path, ids: list[int]) -> dict[int, dict]:
    """
    Read only the records we need from metadata.jsonl, keyed by vector_id.
    Why: metadata.jsonl can grow large; scanning once and filtering by id is simple and robust.
    Side effects: file I/O read only.
    """
    wanted = set(ids)
    out: dict[int, dict] = {}
    p = Path(metadata_path)
    if not p.exists():
        # Be explicit to surface misconfiguration
        raise FileNotFoundError(f"Metadata file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not wanted:
                # Early exit once all ids are found (no break; use guard)
                pass
            try:
                obj = json.loads(line)
            except Exception:
                continue
            vid = obj.get("vector_id")
            if isinstance(vid, int) and vid in wanted and vid not in out:
                out[vid] = obj
                # Remove from wanted when satisfied
                wanted.remove(vid)
    return out


def _embed_query(query_text: str, embedding_model: str) -> np.ndarray:
    """
    Embed the query string with the same model used for ingestion.
    Why: the vector space must match the index vectors.
    Returns a (1, dim) float32 array normalised to unit length.
    """
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")
    vec = create_embeddings([query_text], model_name=embedding_model)[0]
    vec = l2_normalise([vec])[0]
    return np.array([vec], dtype="float32")


def retrieve_top_k_chunks(
    query_text: str,
    index_path: Path,
    metadata_path: Path,
    embedding_model: str,
    k: int = 5,
) -> list[dict]:
    """
    Retrieve top-K anonymised chunks for a query.
    Returns: list of { "vector_id": int, "score": float, "chunk_text": str }
    Notes:
      - 'score' is the FAISS L2 distance; lower is closer. You may convert to similarity if needed.
      - Chunks already contain [FILTERED_*] placeholders when anonymisation was enabled at ingest time.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    index = _load_index(Path(index_path))
    q = _embed_query(query_text, embedding_model)

    # FAISS returns distances (D) and indices (I)
    D, I = index.search(q, k)
    ids = [int(i) for i in I[0] if i >= 0]
    dists = [float(d) for d in D[0][: len(ids)]]

    # Map vector ids to their anonymised chunk texts
    meta_map = _read_metadata_subset(Path(metadata_path), ids)

    results: list[dict] = []
    for i, vid in enumerate(ids):
        # If metadata row is missing (inconsistent state), provide an empty snippet
        chunk_text = ""
        row = meta_map.get(vid)
        if isinstance(row, dict):
            chunk_text = row.get("chunk_text", "") or ""
        results.append({
            "vector_id": vid,
            "score": dists[i],
            "chunk_text": chunk_text,
        })
    return results