from pathlib import Path
from pipeline import process_document_to_faiss
from query import retrieve_top_k_chunks

def test_roundtrip_ingest_and_query(tmp_path, monkeypatch_embeddings):
    doc_path = tmp_path / "sample.txt"
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.jsonl"

    content = (
        "Alice moved to Brussels in 2024 and works at a local bakery.\n"
        "Her favorite pastry is the almond croissant.\n"
        "You can find great bakeries near the Grand Place."
    )
    doc_path.write_text(content, encoding="utf-8")

    count = process_document_to_faiss(
        input_path=doc_path,
        index_path=index_path,
        metadata_path=metadata_path,
        chunk_size=80,
        chunk_overlap=10,
        embedding_model="ignored-by-mock",
        use_llm_privacy_filter=False
    )
    assert count > 0
    assert index_path.exists()
    assert metadata_path.exists()

    hits = retrieve_top_k_chunks(
        query_text="Where are good bakeries in Brussels?",
        k=3,
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model="ignored-by-mock"
    )
    assert len(hits) > 0
    assert "baker" in hits[0]["chunk_text"].lower()
