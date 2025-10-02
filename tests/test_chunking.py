from pipeline import create_overlapping_chunks

def test_create_overlapping_chunks_basic():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = create_overlapping_chunks(text, size=10, overlap=2)
    assert len(chunks) >= 3
    assert chunks[0].startswith("a")
    assert chunks[0].endswith("j")
    assert chunks[0][-2:] == chunks[1][:2]
