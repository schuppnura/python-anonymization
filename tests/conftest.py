import hashlib
import numpy as np
import pytest
import sys
from pathlib import Path

# Add the project root directory to Python path so tests can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def make_deterministic_vector(text: str, dim: int = 1024) -> np.ndarray:
    h = hashlib.sha256(text.encode('utf-8')).digest()
    seed = int.from_bytes(h[:4], 'big', signed=False)
    rng = np.random.RandomState(seed)
    vec = rng.rand(dim).astype('float32')
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm

@pytest.fixture
def monkeypatch_embeddings(monkeypatch):
    import ollama
    def fake_embeddings(model: str, prompt: str):
        # Return a fake embedding response that matches ollama's format
        embedding_vector = make_deterministic_vector(prompt, 1024)
        return {"embedding": embedding_vector.tolist()}
    monkeypatch.setattr(ollama, "embeddings", fake_embeddings)
    return True
