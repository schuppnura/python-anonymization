import faiss
import numpy as np
import json
from pathlib import Path
from typing import Iterable, Tuple
from constants import EMBEDDING_DIMENSION

def build_or_load_index(index_path: Path) -> faiss.Index:
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return faiss.IndexFlatIP(EMBEDDING_DIMENSION)

def add_vectors_to_index(index: faiss.Index, vectors: np.ndarray) -> Tuple[int, int]:
    previous_total = index.ntotal
    index.add(vectors)
    new_total = index.ntotal
    return previous_total, new_total

def write_index_to_disk(index: faiss.Index, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

def append_metadata(metadata_path: Path, records: Iterable[dict]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_metadata_map(metadata_path: Path) -> dict[int, dict]:
    mapping: dict[int, dict] = {}
    if not metadata_path.exists():
        return mapping
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            vector_id = int(obj["vector_id"])
            mapping[vector_id] = obj
    return mapping

def search_top_k(index: faiss.Index, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores, ids = index.search(query_vector, k)
    return scores, ids
