from pathlib import Path
from pypdf import PdfReader
from docx import Document

def read_document_to_text(input_path: Path) -> str:
    suffix = input_path.suffix.lower()
    if suffix == ".txt":
        return read_txt(input_path)
    if suffix == ".pdf":
        return read_pdf(input_path)
    if suffix == ".docx":
        return read_docx(input_path)
    raise ValueError(f"Unsupported file type: {suffix}")

def read_txt(input_path: Path) -> str:
    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read()

def read_pdf(input_path: Path) -> str:
    reader = PdfReader(str(input_path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def read_docx(input_path: Path) -> str:
    doc = Document(str(input_path))
    return "\n".join(p.text for p in doc.paragraphs)
