from pathlib import Path
import hashlib

from document_loader import read_document_to_text
from anonymiser import (
    apply_regex_redaction,
    apply_person_redaction_with_llm,
)
from embedding import create_embeddings, l2_normalise
from faiss_store import (
    build_or_load_index,
    add_vectors_to_index,
    write_index_to_disk,
    append_metadata,
)


def process_document_to_faiss(
    input_path: Path,
    index_path: Path,
    metadata_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    llm_person_redaction_enabled: bool,
    llm_person_model_name: str,
) -> int:
    """
    Ingest a document: read → anonymise (regex → LLM-person → optional contextual)
    → chunk → embed → normalise → index → persist metadata.
    Assumptions: input_path exists; Ollama is running if LLM-person is enabled.
    Side effects: writes FAISS index & metadata JSONL.
    """
    # 1) Load
    text = read_document_to_text(input_path)
    report_path = Path("data") / "redaction_report.json"  # project-local diagnostic

    # 2) Regex PII first (emails/IBAN/phones)
    redacted = apply_regex_redaction(text)

    # 3) LLM-based PERSON redaction (names like 'De heer Carlo Schupp, voornoemd')
    redacted = apply_person_redaction_with_llm(
        redacted,
        model_name=llm_person_model_name,
        enabled=llm_person_redaction_enabled,
        debug_report_path=report_path,
    )

    # 4) Chunk → embed → normalise → index
    chunks = create_overlapping_chunks(redacted, chunk_size, chunk_overlap)
    matrix = create_embeddings(chunks, model_name=embedding_model)
    matrix = l2_normalise(matrix)

    index = build_or_load_index(index_path)
    previous_total, new_total = add_vectors_to_index(index, matrix)
    write_index_to_disk(index, index_path)

    # 6) Persist metadata (NOTE: we store the already-redacted chunk_text)
    records = []
    doc_id = hash_path(input_path)
    for i, chunk in enumerate(chunks):
        vector_id = previous_total + i
        records.append(
            {
                "vector_id": vector_id,
                "document_id": doc_id,
                "source_path": str(input_path),
                "chunk_index": i,
                "chunk_text": chunk,
            }
        )
    append_metadata(metadata_path, records)

    return new_total - previous_total


def create_overlapping_chunks(text: str, size: int, overlap: int) -> list[str]:
    """
    Create overlapping chunks of text; ensures context carryover for retrieval.
    Assumptions: size > 0 and 0 <= overlap < size.
    Side effects: none.
    """
    if size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + size, text_len)
        chunks.append(text[start:end])
        start = start + size - overlap
    return chunks


def hash_path(path: Path) -> str:
    """
    Return a short stable id for a document path.
    Assumptions: path resolves; collisions extremely unlikely for 16 hex chars.
    Side effects: none.
    """
    hasher = hashlib.sha256()
    hasher.update(str(path.resolve()).encode("utf-8"))
    return hasher.hexdigest()[:16]
