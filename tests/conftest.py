import hashlib
import numpy as np
import pytest

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
    import embedding
    def fake_create_embeddings(texts, model_name: str):
        import numpy as np
        vectors = [make_deterministic_vector(t, 1024) for t in texts]
        return np.vstack(vectors).astype('float32')
    monkeypatch.setattr(embedding, "create_embeddings", fake_create_embeddings)
    return True
