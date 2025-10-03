# anonymiser.py
# Self-contained anonymiser + ingest pipeline.
# API: process_text_to_faiss(original_text, primary_lang, source_path, index_path, metadata_path,
#                            chunk_size, chunk_overlap, embedding_model,
#                            llm_person_redaction_enabled, llm_person_model_name) -> int
#
# Design notes:
# - Names spelled out; loop vars may be short.
# - Functions start with verbs; no nested functions.
# - Comments explain why; state assumptions and side effects.
# - No 'break'; refactor control flow.
# - Separate I/O from core logic (this module does no file reads of source docs).
# - Validate inputs early; clear exceptions on failure.

from __future__ import annotations

import hashlib
import json as _json
import math
import re
import string
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# ---------------------------- Constants (single source of truth) ----------------------------

MONTHS: Dict[str, List[str]] = {
    "nl": ["januari","februari","maart","april","mei","juni","juli","augustus","september","oktober","november","december"],
    "fr": ["janvier","février","mars","avril","mai","juin","juillet","août","septembre","octobre","novembre","décembre"],
    "en": ["January","February","March","April","May","June","July","August","September","October","November","December"],
}

STREETS: Dict[str, str] = {
    "nl": r"(?:straat|laan|steenweg|weg|plein|dreef|lei|kaai|baan|hof|markt|vest|ring|kouter|oever)",
    "fr": r"(?:rue|avenue|boulevard|chaussée|place|allée|quai|impasse|cours)",
    "en": r"(?:Street|Avenue|Boulevard|Road|Lane|Drive|Court|Place|Square|Way)",
}

PII_REGEX_PATTERNS: dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"\b(?:\+?\d[\d\s\-]{6,}\d)\b",
    "iban":  r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",
    "be_rrn": r"\b\d{2}[.\-\/]?\d{2}[.\-\/]?\d{2}[-\s]?\d{3}[.\-\/]?\d{2}\b",
}

DATE_NUM_ANY = r"\b[0-3]?\d[./-][0-1]?\d[./-]\d{2,4}\b"
BE_POSTAL_CODE_REGEX = r"\b[1-9]\d{3}\b"

BIRTHPLACE_PHRASE = {
    "nl": r"\bgeboren\s+(?:te|in)\s+[A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*\b",
    "fr": r"\bn[ée]\(?e\)?\s+à\s+[A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*\b",
    "en": r"\bborn\s+(?:in|at)\s+[A-Z][A-Za-z’'`-]*(?:\s+[A-Z][A-Za-z’'`-]*)*\b",
}


def build_month_alternation(lang: str) -> str:
    """Return a regex alternation for month names in the selected language."""
    months = MONTHS.get(lang, [])
    escaped = [re.escape(m) for m in months]
    return "(?:" + "|".join(escaped) + ")"


def build_long_date_regex(lang: str) -> str:
    """Return a regex for long-form dates per language; falls back to match-nothing regex."""
    if lang == "nl":
        return rf"\b([0-3]?\d)\s+{build_month_alternation('nl')}\s+\d{{4}}\b"
    if lang == "fr":
        return rf"\b([0-3]?\d)\s+{build_month_alternation('fr')}\s+\d{{4}}\b"
    if lang == "en":
        m = build_month_alternation('en')
        d1 = rf"\b([0-3]?\d)(?:st|nd|rd|th)?\s+{m},?\s+\d{{4}}\b"
        d2 = rf"\b{m}\s+([0-3]?\d)(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
        return rf"(?:{d1}|{d2})"
    return r"$^"  # match nothing


def build_address_regex(lang: str) -> str:
    """Return a regex capturing common address structures in BE/NL/FR/EN contexts."""
    streets = STREETS.get(lang, r"")
    if lang in ("nl", "en"):
        return (
            rf"\b([A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*\s+{streets}\s+\d+[A-Za-z]?"
            rf"(?:\s*(?:bus|busnr\.?|b\.?|box|unit|apt\.?|suite)\s*\w+)?"
            rf"(?:\s*,\s*{BE_POSTAL_CODE_REGEX}\s+[A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*)?"
            rf")\b"
        )
    if lang == "fr":
        return (
            rf"\b({streets}\s+[A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*\s+\d+[A-Za-z]?"
            rf"(?:\s*(?:bte|boîte|box)\s*\w+)?"
            rf"(?:\s*,\s*{BE_POSTAL_CODE_REGEX}\s+[A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*)?"
            rf")\b"
        )
    return r"$^"  # match nothing


def build_lang_regex_bundle(lang: str) -> dict[str, str]:
    """Bundle language-specific regex patterns so we update one place."""
    return {
        "date_long": build_long_date_regex(lang),
        "date_numeric": DATE_NUM_ANY,
        "birthplace": BIRTHPLACE_PHRASE.get(lang, r"$^"),
        "address": build_address_regex(lang),
    }


# ---------------------------- LLM helpers (local Ollama) ----------------------------

try:
    import ollama
except Exception:
    ollama = None  # graceful degradation if Ollama is not installed


# System prompts (renamed: systemprompt_*)
systemprompt_person = (
    "You are a named-entity recognizer. Identify all PERSON names.\n"
    "Return STRICT JSON only with this schema:\n"
    '{ "persons": [ {"given": ["..."], "family": ["..."]}, ... ] }\n'
    "- Put all given names (in order) in 'given'.\n"
    "- Put all family-name tokens in 'family'.\n"
    "- EXCLUDE titles, roles, and legal qualifiers.\n"
    "- Do not include organizations or locations."
)

systemprompt_spans = (
    "You are a PII finder. Identify spans for sensitive data in the text.\n"
    "Return STRICT JSON:\n"
    '{ "spans": [ {"label":"DATE_OF_BIRTH"|"BIRTHPLACE"|"ADDRESS", "start":int, "end":int, "text":"..."}, ... ] }\n'
    "- DATE_OF_BIRTH: explicit birth dates tied to a birth context.\n"
    "- BIRTHPLACE: locations signaled by phrases like 'geboren te ...', 'né à ...', 'born in ...'.\n"
    "- ADDRESS: postal addresses (street + number, optional unit/box, postal code + locality).\n"
    "Offsets are 0-based char positions in ORIGINAL text; end is exclusive.\n"
    "Output JSON only."
)

# User prompt templates (renamed: userprompt_*)
userprompt_spans_template = (
    "Find DATE_OF_BIRTH, BIRTHPLACE, ADDRESS spans."
    "{lang_hint}"
    " Return JSON only.\nText:\n{text}"
)


def call_ollama_json(model_name: str, systemprompt: str, userprompt: str) -> Dict[str, Any]:
    """Call Ollama chat with enforced JSON output. Assumes local inference host."""
    if ollama is None:
        return {}
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "system", "content": systemprompt}, {"role": "user", "content": userprompt}],
            options={"temperature": 0.0, "format": "json"},
        )
        content = response["message"]["content"]
        try:
            return _json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end > start:
                return _json.loads(content[start:end + 1])
            return {}
    except Exception:
        return {}


# ----- PERSON name detection helpers (tolerant matching without leaking original text) -----

def normalise_compare(s: str) -> str:
    """Normalise a string for tolerant matching: fold accents, strip punctuation, casefold, collapse spaces."""
    s_norm = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s_norm = s_norm.casefold()
    s_norm = s_norm.translate(str.maketrans("", "", string.punctuation))
    s_norm = " ".join(s_norm.split())
    return s_norm


def build_index_map(original: str) -> tuple[str, List[int]]:
    """
    Build a lossy normalised view of the text and an index map back to original positions.
    Why: lets us search robustly (diacritics, punctuation, spacing) yet redact true char spans.
    """
    norm_chars: List[str] = []
    idx_map: List[int] = []
    last_space = False
    i = 0
    while i < len(original):
        ch = original[i]
        d = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
        if len(d) == 0:
            i += 1
            continue
        if d in string.punctuation:
            i += 1
            continue
        if d.isspace():
            if not last_space:
                norm_chars.append(" ")
                idx_map.append(i)
                last_space = True
        else:
            norm_chars.append(d.casefold())
            idx_map.append(i)
            last_space = False
        i += 1
    return "".join(norm_chars).strip(), idx_map


def find_spans_flexible(haystack: str, needle: str) -> List[Tuple[int, int]]:
    """Find spans of needle inside haystack using tolerant normalised matching."""
    if not needle or not needle.strip():
        return []
    norm_hay, map_hay = build_index_map(haystack)
    ndl = normalise_compare(needle)
    spans: List[Tuple[int, int]] = []
    start = 0
    done = False
    while not done:
        idx = norm_hay.find(ndl, start)
        if idx == -1:
            done = True
        else:
            s0 = map_hay[idx]
            end_norm = idx + len(ndl) - 1
            e0 = map_hay[end_norm] + 1 if end_norm < len(map_hay) else len(haystack)
            spans.append((s0, e0))
            start = idx + len(ndl)
            if start >= len(norm_hay):
                done = True
    return spans


def merge_overlapping_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlaps so we do not double-mask."""
    if not spans:
        return spans
    spans_sorted = sorted(spans, key=lambda p: (p[0], p[1]))
    merged: List[Tuple[int, int]] = []
    i = 0
    while i < len(spans_sorted):
        s, e = spans_sorted[i]
        if not merged:
            merged.append((s, e))
        else:
            ls, le = merged[-1]
            if s <= le:
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))
        i += 1
    return merged


def apply_mask(text: str, spans: List[Tuple[int, int]], placeholder: str) -> str:
    """Mask spans in reverse order so indices stay valid."""
    result = text
    i = len(spans) - 1
    while i >= 0:
        s, e = spans[i]
        if e > s:
            result = result[:s] + placeholder + result[e:]
        i -= 1
    return result


def variants_for_person(person: Dict[str, List[str]]) -> List[str]:
    """Generate naive person string variants like 'Given Family' and 'Family Given'."""
    given = " ".join(person.get("given", []))
    family = " ".join(person.get("family", []))
    variants: List[str] = []
    if len(given) > 0 and len(family) > 0:
        variants.append(f"{given} {family}")
        variants.append(f"{family} {given}")
    elif len(family) > 0:
        variants.append(family)
    elif len(given) > 0:
        variants.append(given)

    out: List[str] = []
    seen = set()
    j = 0
    while j < len(variants):
        v = variants[j]
        if v not in seen:
            out.append(v)
            seen.add(v)
        j += 1
    return out


def extract_person_structs_llm(text: str, model_name: str) -> List[Dict[str, List[str]]]:
    """Call LLM to extract persons [{given:[], family:[]}, ...]; exclude titles and roles."""
    userprompt = "Extract PERSON names as {persons:[{given:[],family:[]}]}.\nText:\n" + text
    obj = call_ollama_json(model_name, systemprompt_person, userprompt)
    persons = obj.get("persons", []) if isinstance(obj, dict) else []
    clean: List[Dict[str, List[str]]] = []
    for p in persons:
        if isinstance(p, dict):
            given = [t.strip() for t in p.get("given", []) if isinstance(t, str) and t.strip()]
            family = [t.strip() for t in p.get("family", []) if isinstance(t, str) and t.strip()]
            if len(given) > 0 or len(family) > 0:
                clean.append({"given": given, "family": family})
    return clean


# ---------------------------- Filters (apply_ → filter_, placeholders now [FILTERED_*]) ----------------------------

def filter_identifier_regex(text: str) -> str:
    """
    Filter: identifier_regex
    Why: remove obvious identifiers via patterns (email, phone, IBAN, BE RRN).
    Side effects: none besides text replacement.
    """
    red = text
    for key, pat in PII_REGEX_PATTERNS.items():
        red = re.sub(pat, f"[FILTERED_{key.upper()}]", red, flags=re.IGNORECASE)
    return red


def filter_birth_and_address_regex(text: str, active_langs: Iterable[str]) -> str:
    """
    Filter: birth/address via language-aware regex.
    Why: remove DOB expressions, birthplace anchors, addresses using robust patterns per language.
    Assumption: text language(s) provided by caller; we expect 'nl'/'fr'/'en'.
    """
    red = text
    for lang in active_langs:
        rules = build_lang_regex_bundle(lang)
        for key in ("date_long", "date_numeric", "birthplace", "address"):
            pat = rules.get(key)
            if not pat:
                continue
            red = re.sub(pat, f"[FILTERED_{key.upper()}]", red, flags=re.IGNORECASE)
    return red


def filter_birth_and_address_llm(text: str, model_name: str, enabled: bool, primary_lang: str | None = None) -> str:
    """
    Filter: birth/address via LLM spans.
    Why: catch contextual DOB, birthplace, and address spans the regex might miss.
    """
    if not enabled:
        return text
    if ollama is None:
        return text
    lang_hint = f" Primary document language: '{primary_lang}'." if primary_lang else ""
    userprompt = userprompt_spans_template.format(lang_hint=lang_hint, text=text)
    obj = call_ollama_json(model_name, systemprompt_spans, userprompt)

    spans: List[Dict[str, int]] = []
    if isinstance(obj, dict):
        raw = obj.get("spans", [])
        i = 0
        while i < len(raw or []):
            sp = raw[i]
            lab = sp.get("label")
            s = sp.get("start")
            e = sp.get("end")
            if lab in {"DATE_OF_BIRTH", "BIRTHPLACE", "ADDRESS"} and isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text):
                spans.append({"label": lab, "start": s, "end": e})
            i += 1

    spans.sort(key=lambda x: (x["start"], x["end"]))
    merged: List[Dict[str, int]] = []
    i = 0
    while i < len(spans):
        sp = spans[i]
        if not merged:
            merged.append(sp)
        else:
            last = merged[-1]
            if sp["start"] <= last["end"] and sp["label"] == last["label"]:
                last["end"] = max(last["end"], sp["end"])
            else:
                merged.append(sp)
        i += 1

    result = text
    j = len(merged) - 1
    while j >= 0:
        sp = merged[j]
        placeholder = "[FILTERED_" + sp["label"] + "]"
        result = result[:sp["start"]] + placeholder + result[sp["end"]:]
        j -= 1
    return result


def filter_person_name_llm(text: str, model_name: str, enabled: bool, debug_report_path: Path | None = None) -> str:
    """
    Filter: person_name_llm
    Why: detect person names in flexible ways (order, casing, accents, punctuation) and mask them.
    Side effects: optionally writes a JSON report for QA if debug_report_path is provided.
    """
    if not enabled:
        return text
    if ollama is None:
        return text

    max_block_chars = 2000
    blocks = [text] if len(text) <= max_block_chars else [text[i:i + max_block_chars] for i in range(0, len(text), max_block_chars)]
    reports: List[Dict[str, Any]] = []
    out: List[str] = []

    b_idx = 0
    while b_idx < len(blocks):
        b = blocks[b_idx]
        persons = extract_person_structs_llm(b, model_name)
        spans: List[Tuple[int, int]] = []
        detected: List[str] = []

        p_idx = 0
        while p_idx < len(persons):
            for cand in variants_for_person(persons[p_idx]):
                detected.append(cand)
                spans.extend(find_spans_flexible(b, cand))
            p_idx += 1

        spans = merge_overlapping_spans(spans)
        red_b = apply_mask(b, spans, "[FILTERED_PERSON]")

        remaining = []
        k = 0
        while k < len(detected):
            cand = detected[k]
            if len(find_spans_flexible(red_b, cand)) > 0:
                remaining.append(cand)
            k += 1

        reports.append({
            "block_index": b_idx,
            "persons_struct": persons,
            "detected_variants": detected,
            "spans": spans,
            "remaining_variants_after_mask": remaining,
        })
        out.append(red_b)
        b_idx += 1

    if debug_report_path is not None:
        debug_report_path.parent.mkdir(parents=True, exist_ok=True)
        debug_report_path.write_text(_json.dumps({"blocks": reports}, ensure_ascii=False, indent=2), encoding="utf-8")

    return "".join(out)


# ---------------------------- Embeddings via Ollama ----------------------------

try:
    import ollama as _ollama_for_embed
except Exception:
    _ollama_for_embed = None


def create_embeddings(chunks: list[str], model_name: str) -> list[list[float]]:
    """Create one embedding vector per chunk using the local embedding model."""
    if _ollama_for_embed is None:
        raise RuntimeError("Ollama not available for embeddings. Start Ollama or configure another backend.")
    vectors: list[list[float]] = []
    i = 0
    while i < len(chunks):
        resp = _ollama_for_embed.embeddings(model=model_name, prompt=chunks[i])
        vectors.append(resp["embedding"])
        i += 1
    return vectors


def l2_normalise(matrix: list[list[float]]) -> list[list[float]]:
    """L2-normalise vectors for FAISS (consistent distance scale)."""
    out: list[list[float]] = []
    i = 0
    while i < len(matrix):
        v = matrix[i]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
        i += 1
    return out


# ---------------------------- FAISS persistence (self-contained) ----------------------------

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


def build_or_load_index(index_path: Path):
    """Load an existing FAISS index, or return None to create one later when dimension is known."""
    if faiss is None:
        raise RuntimeError("faiss not installed")
    p = Path(index_path)
    if p.exists():
        return faiss.read_index(str(p))
    return None


def add_vectors_to_index(index, matrix: list[list[float]]) -> tuple[int, int, Any]:
    """Add vectors to the index, creating a new IndexFlatL2 when needed."""
    import numpy as _np
    xb = _np.array(matrix, dtype="float32")
    if index is None:
        dim = xb.shape[1]
        index = faiss.IndexFlatL2(dim)
    previous_total = index.ntotal
    index.add(xb)
    return previous_total, index.ntotal, index


def write_index_to_disk(index, index_path: Path) -> None:
    """Persist the FAISS index to disk; ensures parent folder exists."""
    p = Path(index_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p))


def append_metadata(metadata_path: Path, records: list[dict]) -> None:
    """
    Append chunk metadata as JSON lines.
    Why: reconstruct chunks on query and preserve source and ordering.
    """
    p = Path(metadata_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        i = 0
        while i < len(records):
            f.write(_json.dumps(records[i], ensure_ascii=False) + "\n")
            i += 1


# ---------------------------- Chunking helpers ----------------------------

def create_overlapping_chunks(text: str, size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks for embedding.
    Assumptions: size > 0; 0 <= overlap < size
    """
    if size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        start = start + size - overlap
    return chunks


def hash_path(path: Path) -> str:
    """Stable short document id from absolute path, for metadata grouping."""
    hasher = hashlib.sha256()
    hasher.update(str(Path(path).resolve()).encode("utf-8"))
    return hasher.hexdigest()[:16]


# ---------------------------- Ingest pipeline ----------------------------

def process_text_to_faiss(
    original_text: str,
    primary_lang: str,
    source_path: Path,
    index_path: Path,
    metadata_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    llm_person_redaction_enabled: bool,
    llm_person_model_name: str,
) -> int:
    """
    End-to-end anonymisation + indexing.
    Steps:
      1) filter_identifier_regex              → [FILTERED_EMAIL|PHONE|IBAN|BE_RRN]
      2) filter_birth_and_address_regex       → [FILTERED_DATE_LONG|DATE_NUMERIC|BIRTHPLACE|ADDRESS]
      3) filter_birth_and_address_llm         → [FILTERED_DATE_OF_BIRTH|BIRTHPLACE|ADDRESS]
      4) filter_person_name_llm               → [FILTERED_PERSON]
      5) chunk → embed → L2 normalise → add to FAISS → persist index + metadata
    Side effects:
      - Writes index to index_path and appends metadata to metadata_path.
      - Optionally writes data/redaction_report.json for QA.
    """
    if not isinstance(original_text, str) or len(original_text) == 0:
        raise ValueError("original_text must be a non-empty string")
    if primary_lang not in ("nl", "fr", "en"):
        # Assumption: default to 'nl' in BE context when auto-detection is missing
        primary_lang = "nl"

    active_langs = [primary_lang]

    # 1) Obvious identifiers via regex
    filtered = filter_identifier_regex(original_text)

    # 2) Language-aware regex for DOB terms, birthplace anchors, and addresses
    filtered = filter_birth_and_address_regex(filtered, active_langs)

    # 3) LLM-based DOB/place/address spans (optional)
    filtered = filter_birth_and_address_llm(
        filtered,
        model_name=llm_person_model_name,
        enabled=llm_person_redaction_enabled,
        primary_lang=primary_lang,
    )

    # 4) LLM-based person name masking (optional)
    filtered = filter_person_name_llm(
        filtered,
        model_name=llm_person_model_name,
        enabled=llm_person_redaction_enabled,
        debug_report_path=Path("data") / "redaction_report.json",
    )

    # 5) Chunk → embed → normalise → index → persist
    chunks = create_overlapping_chunks(filtered, chunk_size, chunk_overlap)
    matrix = create_embeddings(chunks, model_name=embedding_model)
    matrix = l2_normalise(matrix)

    index = build_or_load_index(index_path)
    previous_total, new_total, index = add_vectors_to_index(index, matrix)
    write_index_to_disk(index, index_path)

    # Metadata: store filtered chunks only
    records = []
    doc_id = hash_path(source_path)
    i = 0
    while i < len(chunks):
        vector_id = previous_total + i
        records.append({
            "vector_id": vector_id,
            "document_id": doc_id,
            "source_path": str(source_path),
            "chunk_index": i,
            "chunk_text": chunks[i],
        })
        i += 1
    append_metadata(metadata_path, records)

    return new_total - previous_total