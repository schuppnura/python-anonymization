# document_loader.py
# Read documents to text and detect the primary language.
#
# Functions:
#   read_document_to_text(path: Path) -> str
#   detect_language(text: str) -> tuple[str, list[tuple[str, float]]]
#
# Design rules respected:
# - Functions start with verbs; no nested functions
# - Comments explain why; state assumptions and side effects
# - Validate inputs early; clear error messages
# - Separate I/O from core logic; return data, don’t print

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import re


# We keep a minimal month lexicon here to avoid tight coupling and circular imports.
# Why: the heuristic detector uses these as weak signals if langdetect is missing or unsure.
MONTHS = {
    "nl": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"],
    "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
}

SUPPORTED_EXTS = {".txt", ".pdf", ".docx"}

# Optional dependencies (loaded lazily and guarded)
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

try:
    from docx import Document  # type: ignore
except Exception:
    Document = None

try:
    from langdetect import detect_langs  # type: ignore
except Exception:
    detect_langs = None


def read_document_to_text(input_path: Path) -> str:
    """
    Read a document (.txt/.pdf/.docx) to UTF-8 text.

    Assumptions:
    - Caller passes a valid path.
    Side effects:
    - File I/O only.

    Raises:
    - FileNotFoundError on missing path
    - ValueError on unsupported extension
    - RuntimeError if required parser library is missing
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type: {p.suffix}. Supported: {sorted(SUPPORTED_EXTS)}")

    ext = p.suffix.lower()
    if ext == ".txt":
        # Assumption: UTF-8; we allow replacement to avoid hard failure on rare bytes
        return p.read_text(encoding="utf-8", errors="replace")

    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf not installed; cannot read PDFs")
        reader = PdfReader(str(p))
        # Not all PDFs have clean text extraction; we still return best-effort text
        parts: List[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)

    if ext == ".docx":
        if Document is None:
            raise RuntimeError("python-docx not installed; cannot read DOCX")
        doc = Document(str(p))
        return "\n".join(par.text for par in doc.paragraphs)

    # Defensive fallback (should never hit due to SUPPORTED_EXTS guard)
    raise ValueError(f"Unsupported file type encountered unexpectedly: {ext}")


def detect_language(text: str) -> Tuple[str, List[tuple]]:
    """
    Detect the primary document language among ('nl', 'fr', 'en').

    Strategy:
    1) If langdetect is available, use it on the first ~15k chars; pick the highest in our set.
    2) Fallback heuristic: count month-name matches per language and convert to pseudo-probs.

    Returns:
      (primary_lang, scored_list) where primary_lang in {'nl','fr','en'}
      and scored_list is a list of (lang_code, score/probability) sorted descending.

    Assumptions:
    - 'text' is raw document text (can be long).
    Side effects:
    - None.
    """
    if not isinstance(text, str) or not text.strip():
        # Default to 'nl' in BE contexts if text is empty; expose neutral scores
        return "nl", [("nl", 1.0), ("fr", 0.0), ("en", 0.0)]

    candidates = ("nl", "fr", "en")

    # 1) Probabilistic detection when available
    if detect_langs is not None:
        try:
            results = detect_langs(text[:15000])  # cap for speed
            # Map to our candidate set only
            scored = [(r.lang, float(r.prob)) for r in results if r.lang in candidates]
            if scored:
                scored.sort(key=lambda x: x[1], reverse=True)
                primary = scored[0][0]
                return primary, scored
        except Exception:
            # Fall through to heuristic if langdetect fails
            pass

    # 2) Heuristic: month-name hits as weak signals
    hits = {k: 0 for k in candidates}
    for lang in candidates:
        for month in MONTHS[lang]:
            # Case-insensitive whole-word match
            hits[lang] += len(re.findall(rf"\b{re.escape(month)}\b", text, flags=re.IGNORECASE))

    # Derive pseudo-probabilities
    primary_language = max(hits, key=hits.get)
    total = sum(hits.values())
    if total == 0:
        # If no signal at all, prefer 'nl' by convention (BE context); expose neutral scores
        return "nl", [("nl", 1/3), ("fr", 1/3), ("en", 1/3)]

    ranked = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
    probs = [(lang, hits[lang] / total) for lang, _ in ranked]
    return primary_language, probs
