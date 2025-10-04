# document_loader.py
# Reading documents + deterministic language detection (singleton only).

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

# Public libs only; no local imports
import langid           # pip install langid
from docx import Document as DocxDocument  # pip install python-docx
from pypdf import PdfReader               # pip install pypdf


SUPPORTED_LANGS = {"nl", "fr", "en"}
# Map detector outputs/variants to our supported set
LANG_NORMALISATION = {
    "nl": "nl", "nld": "nl", "dut": "nl",
    "fr": "fr", "fra": "fr", "fre": "fr",
    "en": "en", "eng": "en",
}


def read_document_to_text(path_str: str) -> str:
    """
    Read a .docx or .pdf into plain text.
    Why:
        Ingest and redaction work on raw text; keep I/O here, logic elsewhere.
    How:
        - For .docx use python-docx
        - For .pdf  use pypdf
        - For anything else, treat as UTF-8 text
    Side effects:
        None. Raises ValueError for missing/empty content.
    """
    path = Path(path_str)
    if not path.exists():
        raise ValueError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        parts = []
        for p in doc.paragraphs:
            if p.text:
                parts.append(p.text)
        text = "\n".join(parts)
    elif suffix == ".pdf":
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                parts.append(t)
        text = "\n".join(parts)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    text = text.strip()
    if not text:
        raise ValueError(f"No extractable text from: {path}")
    return text


def detect_language(text: str) -> str:
    """
    Detect the primary language and return a SINGLE code: 'nl'|'fr'|'en'.
    Why:
        Downstream anonymiser expects a singleton code (not lists/mixes);
        MCP server and CLI share this behavior.
    How:
        Use langid.classify -> normalize via LANG_NORMALISATION -> fallback to 'nl'
        if the detector gives an unsupported code.
    Side effects:
        None. Deterministic on the given text.
    """
    if not isinstance(text, str) or not text.strip():
        return "nl"

    code, _ = langid.classify(text)  # returns e.g. ('nl', -45.12)
    code = code.lower()
    normalised = LANG_NORMALISATION.get(code)
    if normalised in SUPPORTED_LANGS:
        return normalised
    # If detector returns something else, default to 'nl' to stay predictable
    return "nl"
