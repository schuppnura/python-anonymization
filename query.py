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

    Why: FAISS must be reloaded for each process that queries it.
    How: verify the path exists and call faiss.read_index.
    """
    p = Path(index_path)
    if not p.exists():
        raise FileNotFoundError(f"FAISS index not found at {p}")
    return faiss.read_index(str(p))


def load_metadata(metadata_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load chunk metadata keyed by vector_id.

    Why: metadata JSONL provides provenance for results.
    How: read each JSON line; map by vector_id; skip malformed lines.
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
    Create an embedding vector for the query text.

    Why: must match the embedding model used at ingest time.
    How: call ollama.embeddings(model=..., prompt=text).
    """
    resp = ollama.embeddings(model=model_name, prompt=text)
    return resp["embedding"]


def l2_normalise(vec: List[float]) -> List[float]:
    """
    L2-normalise one vector.

    Why: ensures distance magnitudes are comparable in IndexFlatL2.
    How: divide by Euclidean norm (guard zero with 1.0).
    """
    import math
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def query_index(
    query_text: str,
    index_path: Path,
    metadata_path: Path,
    embedding_model: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Query FAISS with a natural-language string and return top-k hits.

    Why: retrieve anonymised chunks most relevant to the query.
    How:
        - Embed and L2-normalise the query
        - FAISS search
        - Join ids back to metadata
    Side effects: none (read-only).
    """
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")

    index = load_index(index_path)
    meta_by_id = load_metadata(metadata_path)

    q = l2_normalise(embed_text(query_text, embedding_model))
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