import numpy as np
import ollama
from typing import Sequence

def create_embeddings(texts: Sequence[str], model_name: str) -> np.ndarray:
    vectors: list[list[float]] = []
    for t in texts:
        response = ollama.embeddings(model=model_name, prompt=t)
        vectors.append(response["embedding"])
    return np.array(vectors, dtype="float32")

def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe_norms
