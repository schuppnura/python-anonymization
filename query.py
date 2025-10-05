# query.py
# Query the FAISS index and return filtered chunks with distances.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import faiss  # type: ignore
import numpy as np
import ollama

logger = logging.getLogger("query")


def load_index(index_path: Path):
    """
    Load a FAISS index from disk.

    Why:
        FAISS indices are persisted by ingest and must be reloaded for querying.
    How:
        Read from the configured path; raise if missing.
    """
    p = Path(index_path)
    if not p.exists():
        raise FileNotFoundError(f"FAISS index not found at {p}")
    return faiss.read_index(str(p))


def load_metadata(metadata_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load chunk metadata keyed by vector_id.

    Why:
        Retrieval needs to map FAISS ids back to chunk text and provenance.
    How:
        Read JSONL lines and index by vector_id; skip malformed lines with a warning.
    """
    p = Path(metadata_path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata JSONL not found at {p}")
    by_id: Dict[int, Dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                vid = int(obj["vector_id"])
                by_id[vid] = obj
            except Exception as e:
                logger.warning("Skipping bad metadata line: %s (%s)", line[:120], e)
    return by_id


def embed_text(text: str, model_name: str) -> List[float]:
    """
    Create an embedding vector for the query.

    Why:
        We must use the same embedding model as ingest to ensure vector space compatibility.
    How:
        Call ollama.embeddings and return the 'embedding' field.
    """
    resp = ollama.embeddings(model=model_name, prompt=text)
    return resp["embedding"]


def l2_normalise(vec: List[float]) -> List[float]:
    """
    L2-normalise one vector.

    Why:
        IndexFlatL2 expects comparable magnitudes for distance ranking.
    How:
        Divide by Euclidean norm (guard zero with 1.0).
    """
    import math
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def query_index(
    question: str,
    index_path: Path,
    metadata_path: Path,
    embedding_model: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Query FAISS with a natural-language string and return top_k hits.

    Why:
        Retrieve the most relevant anonymised chunks for downstream LLM answering.
    How:
        - Embed and L2-normalise the query
        - FAISS search (top_k)
        - Join ids back to metadata
    Side effects:
        None (read-only).
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    index = load_index(index_path)
    meta_by_id = load_metadata(metadata_path)

    q = l2_normalise(embed_text(question, embedding_model))
    qx = np.array([q], dtype="float32")
    k = max(1, int(top_k))

    distances, ids = index.search(qx, k)

    results: List[Dict[str, Any]] = []
    for dist, vid in zip(distances[0].tolist(), ids[0].tolist()):
        if vid == -1:
            continue
        meta = meta_by_id.get(int(vid), {})
        results.append({
            "vector_id": int(vid),
            "distance": float(dist),
            "chunk_text": meta.get("chunk_text", ""),
            "source_path": meta.get("source_path", ""),
            "document_id": meta.get("document_id", ""),
            "chunk_index": meta.get("chunk_index", -1),
        })
    return results
