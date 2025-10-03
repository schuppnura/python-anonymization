# document_loader.py
from pathlib import Path
from typing import Union
from pypdf import PdfReader
from docx import Document

def read_document_to_text(input_path: Union[str, Path]) -> str:
    input_path = Path(input_path)  # <- coerce here
    suffix = input_path.suffix.lower()
    if suffix == ".txt":
        return read_txt(input_path)
    if suffix == ".pdf":
        return read_pdf(input_path)
    if suffix == ".docx":
        return read_docx(input_path)
    raise ValueError(f"Unsupported file type: {suffix}")
    
def read_txt(input_path: Path) -> str:
    return input_path.read_text(encoding="utf-8", errors="replace")

def read_pdf(input_path: Path) -> str:
    reader = PdfReader(str(input_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)

def read_docx(input_path: Path) -> str:
    doc = Document(str(input_path))
    return "\n".join(p.text for p in doc.paragraphs)
