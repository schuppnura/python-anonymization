from pathlib import Path
from embedding import create_embeddings, l2_normalise
from faiss_store import build_or_load_index, load_metadata_map, search_top_k

def retrieve_top_k_chunks(query_text: str, k: int, index_path: Path, metadata_path: Path, embedding_model: str) -> list[dict]:
    index = build_or_load_index(index_path)
    meta = load_metadata_map(metadata_path)
    query_vec = create_embeddings([query_text], model_name=embedding_model)
    query_vec = l2_normalise(query_vec)
    scores, ids = search_top_k(index, query_vec, k)
    hits: list[dict] = []
    for rank, vector_id in enumerate(ids[0].tolist()):
        if vector_id == -1:
            continue
        record = meta.get(int(vector_id))
        if record is None:
            continue
        hits.append({
            "rank": rank + 1,
            "score": float(scores[0][rank]),
            **record
        })
    return hits
