# anonymiser.py
# Self-contained anonymiser + ingest pipeline using span-based, anchor-aware masking.
# Why: provide a single module that can redact PII and persist RAG artifacts without leaking raw PII.
# How: detect spans via regex + LLM prompts; merge with priority; mask in one pass; sentence-aware chunking; FAISS persistence.

from __future__ import annotations

import hashlib
import json
import math
import re
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

# Fail fast (no lazy imports)
import ollama
import faiss  # type: ignore
import numpy as np


# ============================ CONSTANTS (UPPERCASE) ============================

# Identifier regexes (fast, deterministic)
PII_IDENTIFIER_PATTERNS: Dict[str, str] = {
    # Belgian RRN (rijksregisternummer), allows optional separators and spaces around the dash
    "BE_RRN": r"\b\d{2}[.\-/]?\d{2}[.\-/]?\d{2}\s*-\s*\d{3}[.\-/]?\d{2}\b",

    # Email addresses (basic but robust)
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

    # IBAN (up to 34 alphanumeric chars)
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",

    # Phone numbers: allow +, spaces, dashes, min length enforced
    "PHONE": r"\b(?:\+?\d[\d\s\-]{6,}\d)\b",
}

DATE_NUMERIC_ANY = r"\b[0-3]?\d[./-][0-1]?\d[./-]\d{2,4}\b"
BE_POSTAL_CODE_REGEX = r"\b[1-9]\d{3}\b"

MONTHS: Dict[str, List[str]] = {
    "nl": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"],
    "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
}

STREETS: Dict[str, str] = {
    "nl": r"(?:straat|laan|steenweg|weg|plein|dreef|lei|kaai|baan|hof|markt|vest|ring|kouter|oever)",
    "fr": r"(?:rue|avenue|boulevard|chaussée|place|allée|quai|impasse|cours)",
    "en": r"(?:Street|Avenue|Boulevard|Road|Lane|Drive|Court|Place|Square|Way)",
}

# Birthplace anchors: group(1) captures only LOCATION (anchors remain visible)
BIRTHPLACE_PHRASE = {
    "nl": r"\bgeboren\s+(?:te|in)\s+([A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*)",
    "fr": r"\bn[ée]\(?e\)?\s+à\s+([A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*)",
    "en": r"\bborn\s+(?:in|at)\s+([A-Z][A-Za-z’'`-]*(?:\s+[A-Z][A-Za-z’'`-]*)*)",
}

# Residence anchors: group(1) captures the locality
RESIDENCE_PHRASE = {
    "nl": re.compile(r"\b(?:wonende|woonachtig|verblijvend)\s+te\s+([A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*)", re.IGNORECASE),
    "fr": re.compile(r"\b(?:domicilié(?:e)?|demeurant)\s+à\s+([A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*)", re.IGNORECASE),
    "en": re.compile(r"\b(?:residing|domiciled|living)\s+in\s+([A-Z][A-Za-z’'`-]*(?:\s+[A-Z][A-Za-z’'`-]*)*)", re.IGNORECASE),
}

# Honorific anchors: language -> list of compiled patterns; group(1) is the NAME tokens
HONORIFIC_PATTERNS: Dict[str, List[re.Pattern]] = {
    "nl": [
        re.compile(r"\bDe\s+heer\s+((?:[A-ZÀ-Ý][\w’'`-]{2,}\s+){0,5}[A-ZÀ-Ý][\w’'`-]{2,})", re.IGNORECASE),
        re.compile(r"\bMevrouw\s+((?:[A-ZÀ-Ý][\w’'`-]{2,}\s+){0,5}[A-ZÀ-Ý][\w’'`-]{2,})", re.IGNORECASE),
        re.compile(r"\bDhr\.?\s+((?:[A-ZÀ-Ý][\w’'`-]{2,}\s+){0,5}[A-ZÀ-Ý][\w’'`-]{2,})", re.IGNORECASE),
        re.compile(r"\bMevr\.?\s+((?:[A-ZÀ-Ý][\w’'`-]{2,}\s+){0,5}[A-ZÀ-Ý][\w’'`-]{2,})", re.IGNORECASE),
    ],
    "fr": [
        re.compile(r"\bMonsieur\s+((?:[A-ZÀ-Ý][\w’'`-]{2,}\s+){0,5}[A-ZÀ-Ý][\w’'`-]{2,})", re.IGNORECASE),
        re.compile(r"\bMadame\s+((?:[A-ZÀ-Ý][\w’'`-]{2,}\s+){0,5}[A-ZÀ-Ý][\w’'`-]{2,})", re.IGNORECASE),
        re.compile(r"\bM\.?\s+((?:[A-ZÀ-Ý][\w’'`-]{2,}\s+){0,5}[A-ZÀ-Ý][\w’'`-]{2,})", re.IGNORECASE),
        re.compile(r"\bMme\.?\s+((?:[A-ZÀ-Ý][\w’'`-]{2,}\s+){0,5}[A-ZÀ-Ý][\w’'`-]{2,})", re.IGNORECASE),
    ],
    "en": [
        re.compile(r"\bMr\.?\s+((?:[A-Z][A-Za-z’'`-]{2,}\s+){0,5}[A-Z][A-Za-z’'`-]{2,})", re.IGNORECASE),
        re.compile(r"\bMrs\.?\s+((?:[A-Z][A-Za-z’'`-]{2,}\s+){0,5}[A-Z][A-Za-z’'`-]{2,})", re.IGNORECASE),
        re.compile(r"\bMs\.?\s+((?:[A-Z][A-Za-z’'`-]{2,}\s+){0,5}[A-Z][A-Za-z’'`-]{2,})", re.IGNORECASE),
    ],
}

# Label priority in merges (higher wins)
LABEL_PRIORITY: Dict[str, int] = {
    "PERSON": 100,
    "DATE_OF_BIRTH": 90,
    "BIRTHPLACE": 85,
    "ADDRESS": 80,
    "IBAN": 76,       # ensure IBAN beats PHONE if overlaps
    "EMAIL": 72,
    "PHONE": 70,
    "BE_RRN": 70,
    "DATE_LONG": 60,
    "DATE_NUMERIC": 55,
}

# Characters to strip from span edges
PUNCT_STRIP = ",.;:!?()[]{}<>“”‘’\"-–—/\\"

# Prompts (strict JSON; braces escaped for .format)
SYSTEMPROMPT_PERSON = (
    "You are a named-entity recognizer. Identify all PERSON names.\n"
    "Return STRICT JSON only with this schema:\n"
    '{ "persons": [ {"given": ["..."], "family": ["..."]}, ... ] }\n'
    "- Put all given names (in order) in 'given'.\n"
    "- Put all family-name tokens in 'family'.\n"
    "- EXCLUDE titles, roles, and legal qualifiers.\n"
    "- Do not include organizations or locations."
)

BIRTH_ADDR_SYSTEMPROMPT = (
    "You detect privacy-sensitive PII spans in text and MUST return valid JSON.\n"
    "Output requirements:\n"
    '1) Return JSON ONLY (no markdown, no code fences, no natural language).\n'
    '2) The top-level object MUST have exactly these keys: "DATE_OF_BIRTH", "BIRTHPLACE", "ADDRESS".\n'
    '3) Each key maps to a list of objects: {"start": <int>, "end": <int>} (0-based, end exclusive).\n'
    "4) Indices must be valid for the ORIGINAL text.\n"
)

BIRTH_ADDR_USERPROMPT_TEMPLATE = (
    "Find the character spans for the following labels in the text: DATE_OF_BIRTH, BIRTHPLACE, ADDRESS."
    "{lang_hint}\n"
    "Return JSON only, exactly in this structural form (keys and field names EXACTLY as shown):\n"
    '{{"DATE_OF_BIRTH":[{{"start":0,"end":1}}],"BIRTHPLACE":[{{"start":0,"end":1}}],"ADDRESS":[{{"start":0,"end":1}}]}}\n'
    "If none found for a label, return an empty list for that label.\n"
    "Text:\n{text}"
)

# Abbreviations and sentence splitting helpers
ABBREVIATIONS = {
    "nl": {"dhr.", "mevr.", "mej.", "mw.", "mvr.", "dr.", "prof.", "mr.", "ir.", "ing.", "bv.", "nv.", "art.", "pag.", "blz."},
    "fr": {"m.", "mme.", "mlle.", "dr.", "pr.", "prof.", "av.", "env.", "etc.", "p.ex.", "art.", "n°.", "cf."},
    "en": {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "etc.", "e.g.", "i.e.", "no.", "art.", "cf."},
}
QUOTE_CLOSE = set("»›’”\"")


# ============================ Data model ============================

@dataclass(frozen=True)
class MaskSpan:
    """
    Represent a sensitive span.
    Why: merging and masking need a stable structure with hashability.
    How: store start, end (exclusive), and a label string.
    Assumptions: 0 <= start < end <= len(text) at the time of application.
    """
    start: int
    end: int
    label: str


# ============================ Utilities ============================

def build_month_alternation(language: str) -> str:
    """
    Build an alternation of month names per language.
    Why: long-date formats depend on language; centralising avoids duplication.
    How: escape month names and join with '|'.
    """
    months = MONTHS.get(language, [])
    escaped = [re.escape(m) for m in months]
    return "(?:" + "|".join(escaped) + ")"

def build_long_date_regex(language: str) -> str:
    """
    Build a language-aware long date regex.
    Why: support '2 mei 1980', '14 juillet 1975', 'March 5, 1970'/'5 March 1970'.
    How: compose day + month + year; English supports both orders.
    """
    if language == "nl":
        return rf"\b([0-3]?\d)\s+{build_month_alternation('nl')}\s+\d{{4}}\b"
    if language == "fr":
        return rf"\b([0-3]?\d)\s+{build_month_alternation('fr')}\s+\d{{4}}\b"
    if language == "en":
        m = build_month_alternation('en')
        d1 = rf"\b([0-3]?\d)(?:st|nd|rd|th)?\s+{m},?\s+\d{{4}}\b"
        d2 = rf"\b{m}\s+([0-3]?\d)(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
        return rf"(?:{d1}|{d2})"
    return r"$^"

def build_address_regex(language: str) -> str:
    """
    Build a conservative postal address regex.
    Why: match 'StreetType + number' and optionally unit + 'postal code + locality'.
    How: use language-specific street keywords and a BE postal code heuristic when present.
    """
    streets = STREETS.get(language, r"")
    if language in ("nl", "en"):
        return (
            rf"\b([A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*\s+{streets}\s+\d+[A-Za-z]?"
            rf"(?:\s*(?:bus|busnr\.?|b\.?|box|unit|apt\.?|suite)\s*\w+)?"
            rf"(?:\s*,\s*{BE_POSTAL_CODE_REGEX}\s+[A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*)?"
            rf")\b"
        )
    if language == "fr":
        return (
            rf"\b({streets}\s+[A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*\s+\d+[A-Za-z]?"
            rf"(?:\s*(?:bte|boîte|box)\s*\w+)?"
            rf"(?:\s*,\s*{BE_POSTAL_CODE_REGEX}\s+[A-ZÀ-Ý][\w’'`-]*(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]*)*)?"
            rf")\b"
        )
    return r"$^"

def normalize_for_compare(s: str) -> str:
    """
    Normalise a string for tolerant comparison.
    Why: match names regardless of case/accents/punctuation; keep token order simple.
    How: NFKD fold accents; remove punctuation; collapse whitespace; casefold.
    """
    s_norm = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s_norm = s_norm.casefold()
    s_norm = s_norm.translate(str.maketrans("", "", string.punctuation))
    s_norm = " ".join(s_norm.split())
    return s_norm

def build_index_map(original: str) -> Tuple[str, List[int]]:
    """
    Build a normalised surrogate and index map back to original.
    Why: search in normalised space but convert back to original indices.
    How: stream through chars; store original indices for each kept char.
    """
    norm_chars: List[str] = []
    idx_map: List[int] = []
    last_space = False
    i = 0
    while i < len(original):
        ch = original[i]
        d = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
        if len(d) == 0 or d in string.punctuation:
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
    """
    Find approximate spans of a needle in a haystack using the normalised view.
    Why: catch 'Given Family' variants regardless of casing/diacritics.
    How: search normalised haystack, then map indices back to original string.
    """
    if not needle or not needle.strip():
        return []
    norm_hay, map_hay = build_index_map(haystack)
    ndl = normalize_for_compare(needle)
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = norm_hay.find(ndl, start)
        if idx == -1:
            break
        s0 = map_hay[idx]
        end_norm = idx + len(ndl) - 1
        e0 = map_hay[end_norm] + 1 if end_norm < len(map_hay) else len(haystack)
        spans.append((s0, e0))
        start = idx + len(ndl)
        if start >= len(norm_hay):
            break
    return spans

def trim_span_to_tokens(text: str, start: int, end: int) -> Tuple[int, int]:
    """
    Trim whitespace and punctuation from span edges.
    Why: LLM/fuzzy matches can include commas/spaces; trimming keeps anchors intact.
    How: shave leading/trailing whitespace and PUNCT_STRIP; never expand span.
    """
    s, e = start, end
    n = len(text)
    while s < e and s < n and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    while s < e and text[s] in PUNCT_STRIP:
        s += 1
    while e > s and text[e - 1] in PUNCT_STRIP:
        e -= 1
    return s, e


# ============================ Masking and merging ============================

def merge_mask_spans(spans: List[MaskSpan]) -> List[MaskSpan]:
    """
    Merge overlapping spans while keeping the highest-priority label.
    Why: multiple detectors may overlap; merging ensures one placeholder and readable output.
    How: sort by (start,end); merge only when ranges strictly overlap (no touching merge).
    """
    if not spans:
        return []
    spans_sorted = sorted(spans, key=lambda s: (s.start, s.end))
    merged: List[MaskSpan] = []
    for sp in spans_sorted:
        if not merged:
            merged.append(MaskSpan(sp.start, sp.end, sp.label))
            continue
        last = merged[-1]
        if sp.start < last.end:  # strict overlap only
            new_end = max(last.end, sp.end)
            last_pri = LABEL_PRIORITY.get(last.label, 0)
            sp_pri = LABEL_PRIORITY.get(sp.label, 0)
            new_label = last.label if last_pri >= sp_pri else sp.label
            merged[-1] = MaskSpan(last.start, new_end, new_label)
        else:
            merged.append(MaskSpan(sp.start, sp.end, sp.label))
    return merged

def apply_mask_spans(text: str, spans: List[MaskSpan]) -> str:
    """
    Apply placeholders in a single left-to-right pass.
    Why: single render avoids index drift and nested/mangled placeholders.
    How: assemble segments: untouched text + [FILTERED_LABEL] for each span, then tail.
    """
    if not spans:
        return text
    parts: List[str] = []
    cursor = 0
    for sp in sorted(spans, key=lambda s: (s.start, s.end)):
        if sp.start < cursor:
            continue
        parts.append(text[cursor:sp.start])
        parts.append(f"[FILTERED_{sp.label}]")
        cursor = sp.end
    parts.append(text[cursor:])
    return "".join(parts)


# ============================ LLM helpers ============================

def call_ollama_json(model_name: str, systemprompt: str, userprompt: str) -> Dict[str, Any]:
    """
    Call a local LLM via Ollama and parse JSON.
    Why: local inference keeps raw PII on your infrastructure; JSON format is deterministic.
    How: set temperature=0.0 and format='json'; parse strictly with a minimal fallback.
    Assumptions: Ollama is running and the model is pulled; responses include message.content.
    Side effects: network call to the local Ollama server.
    """
    resp = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": userprompt},
        ],
        options={"temperature": 0.0, "format": "json"},
    )
    content = resp["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            return json.loads(content[start:end + 1])
        return {}


# ============================ Span detectors ============================

def detect_identifier_regex_spans(text: str) -> List[MaskSpan]:
    """
    Detect classic identifiers (email, phone, IBAN, BE RRN) via regex.
    Why: precise, fast matches for fixed-format identifiers.
    How: iterate patterns; append a span per match.
    """
    spans: List[MaskSpan] = []
    for label in ["IBAN", "EMAIL", "PHONE", "BE_RRN"]:
        pat = PII_IDENTIFIER_PATTERNS[label]
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            spans.append(MaskSpan(m.start(), m.end(), label))
    return spans

def detect_birth_and_address_regex_spans(text: str, languages: Iterable[str]) -> List[MaskSpan]:
    """
    Detect birthplace (location only), long/numeric dates, and full postal addresses.
    Why: language-aware regex is reliable for common phrasing and formats.
    How: birthplace captures group(1) only; dates and addresses use the full match span.
    """
    spans: List[MaskSpan] = []
    for language in languages:
        rules = {
            "DATE_LONG": build_long_date_regex(language),
            "DATE_NUMERIC": DATE_NUMERIC_ANY,
            "BIRTHPLACE": BIRTHPLACE_PHRASE.get(language, r"$^"),
            "ADDRESS": build_address_regex(language),
        }

        for m in re.finditer(rules["DATE_LONG"], text, flags=re.IGNORECASE):
            spans.append(MaskSpan(m.start(), m.end(), "DATE_LONG"))

        for m in re.finditer(rules["DATE_NUMERIC"], text, flags=re.IGNORECASE):
            spans.append(MaskSpan(m.start(), m.end(), "DATE_NUMERIC"))

        for m in re.finditer(rules["BIRTHPLACE"], text, flags=re.IGNORECASE):
            if m.lastindex and m.lastindex >= 1:
                s, e = m.span(1)
                spans.append(MaskSpan(s, e, "BIRTHPLACE"))

        for m in re.finditer(rules["ADDRESS"], text, flags=re.IGNORECASE):
            spans.append(MaskSpan(m.start(), m.end(), "ADDRESS"))
    return spans

def detect_birthdate_context_spans(text: str, language: str) -> List[MaskSpan]:
    """
    Detect birthdates in explicit context and mask only the date segment.
    Why: keeps anchors like 'geboren op' / 'né le' / 'born on' visible.
    How: language-specific regex with the date in group(1).
    """
    spans: List[MaskSpan] = []
    if language == "nl":
        long_pat = rf"\bgeboren\s+op\s+({build_long_date_regex('nl')})"
        num_pat = rf"\bgeboren\s+op\s+({DATE_NUMERIC_ANY})"
    elif language == "fr":
        long_pat = rf"\bn[ée]\(?e\)?\s+le\s+({build_long_date_regex('fr')})"
        num_pat = rf"\bn[ée]\(?e\)?\s+le\s+({DATE_NUMERIC_ANY})"
    else:
        long_pat = rf"\bborn\s+on\s+({build_long_date_regex('en')})"
        num_pat = rf"\bborn\s+on\s+({DATE_NUMERIC_ANY})"

    for m in re.finditer(long_pat, text, flags=re.IGNORECASE):
        s, e = m.span(1)
        spans.append(MaskSpan(s, e, "DATE_OF_BIRTH"))
    for m in re.finditer(num_pat, text, flags=re.IGNORECASE):
        s, e = m.span(1)
        spans.append(MaskSpan(s, e, "DATE_OF_BIRTH"))
    return spans

def detect_residence_city_spans(text: str, language: str) -> List[MaskSpan]:
    """
    Detect residence anchors like 'wonende te Gent' and mask the city only.
    Why: residence locality is PII even without a full street address.
    How: language-specific regex; group(1) captures the location tokens.
    """
    spans: List[MaskSpan] = []
    pat = RESIDENCE_PHRASE.get(language)
    if isinstance(pat, re.Pattern):
        for m in pat.finditer(text):
            s, e = m.span(1)
            spans.append(MaskSpan(s, e, "ADDRESS"))
    return spans

def extract_birth_address_spans_llm(text: str, model_name: str, language: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Ask the local LLM (via call_ollama_json) to return character spans for DATE_OF_BIRTH, BIRTHPLACE, ADDRESS.
    Why: contextual PII (e.g., 'geboren te ... op ...') is hard to capture with regex only.
    How: strict JSON prompt; expect lists of {'start','end'} per label; validate indices.
    Assumptions: strict prompt compliance; fallback to empty lists if parsing fails.
    Side effects: network call to local model.
    """
    lang_hint = f" Language: {language}." if language else ""
    userprompt = BIRTH_ADDR_USERPROMPT_TEMPLATE.format(lang_hint=lang_hint, text=text)
    raw = call_ollama_json(model_name, BIRTH_ADDR_SYSTEMPROMPT, userprompt)

    out: Dict[str, List[Tuple[int, int]]] = {"DATE_OF_BIRTH": [], "BIRTHPLACE": [], "ADDRESS": []}
    if not isinstance(raw, dict):
        return out

    for label in out.keys():
        arr = raw.get(label, [])
        if not isinstance(arr, list):
            continue
        for item in arr:
            if isinstance(item, dict):
                s, e = item.get("start"), item.get("end")
                if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text):
                    out[label].append((s, e))
    return out

def detect_birth_and_address_llm_spans(text: str, model_name: str, language: str) -> List[MaskSpan]:
    """
    Convert LLM span output into a flat List[MaskSpan] for the merger.
    Why: filter_all expects homogeneous MaskSpan objects across detectors.
    How: call extract_birth_address_spans_llm and map (start,end) to MaskSpan(label).
    """
    spans_by_label = extract_birth_address_spans_llm(text, model_name, language)
    spans: List[MaskSpan] = []
    for label in ("DATE_OF_BIRTH", "BIRTHPLACE", "ADDRESS"):
        for s, e in spans_by_label.get(label, []):
            s2, e2 = trim_span_to_tokens(text, s, e)
            if e2 > s2:
                spans.append(MaskSpan(s2, e2, label))
    return spans

def extract_person_structs_llm(text: str, model_name: str) -> List[Dict[str, List[str]]]:
    """
    Ask the LLM to emit person name structures.
    Why: regex for names is brittle; LLM provides tokenised given/family parts.
    How: request a tiny schema; sanitise to lists of strings; ignore empties.
    """
    userprompt = "Extract PERSON names as {\"persons\":[{\"given\":[],\"family\":[]}]}.\nText:\n" + text
    obj = call_ollama_json(model_name, SYSTEMPROMPT_PERSON, userprompt)
    persons = obj.get("persons", []) if isinstance(obj, dict) else []
    result: List[Dict[str, List[str]]] = []
    for p in persons:
        if isinstance(p, dict):
            given = [t.strip() for t in p.get("given", []) if isinstance(t, str) and t.strip()]
            family = [t.strip() for t in p.get("family", []) if isinstance(t, str) and t.strip()]
            if given or family:
                result.append({"given": given, "family": family})
    return result

def generate_variants_for_person(person: Dict[str, List[str]]) -> List[str]:
    """
    Generate simple surface-form variants for matching.
    Why: cover 'Given Family' and 'Family Given' orders without overfitting.
    How: join tokens in two orders; deduplicate.
    """
    given = " ".join(person.get("given", []))
    family = " ".join(person.get("family", []))
    variants: List[str] = []
    if given and family:
        variants.extend([f"{given} {family}", f"{family} {given}"])
    elif family:
        variants.append(family)
    elif given:
        variants.append(given)
    dedup: List[str] = []
    seen = set()
    for v in variants:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup

def detect_person_name_llm_spans(text: str, model_name: str, language: str) -> List[MaskSpan]:
    """
    Detect person names via LLM + honorific anchors (language-aware).
    Why: names vary; combine LLM candidates with honorific-based captures per language.
    How: get person structs from LLM, generate variants, fuzzy-match; then add spans
         from language-specific honorific patterns that capture the name that follows.
    """
    spans: List[MaskSpan] = []

    persons = extract_person_structs_llm(text, model_name)
    for p in persons:
        for cand in generate_variants_for_person(p):
            for s, e in find_spans_flexible(text, cand):
                s2, e2 = trim_span_to_tokens(text, s, e)
                if e2 > s2:
                    spans.append(MaskSpan(s2, e2, "PERSON"))

    patterns = HONORIFIC_PATTERNS.get(language, [])
    for pat in patterns:
        for m in pat.finditer(text):
            s, e = m.span(1)  # captured name after the honorific
            s2, e2 = trim_span_to_tokens(text, s, e)
            if e2 > s2:
                spans.append(MaskSpan(s2, e2, "PERSON"))

    return spans

def detect_final_safety_spans(text: str) -> List[MaskSpan]:
    """
    Run a final conservative safety sweep.
    Why: defense-in-depth to catch any stragglers (identifiers, numeric dates, generic address-like phrases).
    How: re-run identifier regexes (IBAN first), numeric dates, and a generic address pattern.
    """
    spans: List[MaskSpan] = []
    for label in ["IBAN", "EMAIL", "PHONE", "BE_RRN"]:
        pat = PII_IDENTIFIER_PATTERNS[label]
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            spans.append(MaskSpan(m.start(), m.end(), label))
    for m in re.finditer(DATE_NUMERIC_ANY, text, flags=re.IGNORECASE):
        spans.append(MaskSpan(m.start(), m.end(), "DATE_NUMERIC"))
    generic_address = r"\b\d{1,4}\s+[A-ZÀ-Ý][\w’'`-]+(?:\s+[A-ZÀ-Ýa-zà-ÿ][\w’'`-]+)*\b"
    for m in re.finditer(generic_address, text, flags=re.IGNORECASE):
        spans.append(MaskSpan(m.start(), m.end(), "ADDRESS"))
    return spans


# ============================ Canonical pipeline ============================

def filter_all(
    original_text: str,
    language: str,
    llm_pii_model_name: str,
    debug_report_path: Optional[Path] = None,
) -> str:
    """
    Run the full anonymisation pipeline and return masked text.
    Why: keep all detection on ORIGINAL text and render once to avoid nested placeholders.
    How: combine regex + LLM + residence + safety; merge with priority; apply placeholders.
    Side effects: optionally writes a span report to debug_report_path for QA.
    """
    if not isinstance(original_text, str) or not original_text:
        raise ValueError("original_text must be a non-empty string")
    primary_lang = language if language in ("nl", "fr", "en") else "nl"

    spans: List[MaskSpan] = []
    spans.extend(detect_identifier_regex_spans(original_text))
    spans.extend(detect_birth_and_address_regex_spans(original_text, [primary_lang]))
    spans.extend(detect_residence_city_spans(original_text, primary_lang))
    spans.extend(detect_birthdate_context_spans(original_text, primary_lang))
    spans.extend(detect_birth_and_address_llm_spans(original_text, llm_pii_model_name, primary_lang))
    spans.extend(detect_person_name_llm_spans(original_text, llm_pii_model_name, primary_lang))
    spans.extend(detect_final_safety_spans(original_text))

    merged = merge_mask_spans(spans)

    if debug_report_path is not None:
        debug_report_path.parent.mkdir(parents=True, exist_ok=True)
        report = [{"start": s.start, "end": s.end, "label": s.label, "text": original_text[s.start:s.end]} for s in merged]
        debug_report_path.write_text(json.dumps({"spans": report}, ensure_ascii=False, indent=2), encoding="utf-8")

    return apply_mask_spans(original_text, merged)


# ============================ Embedding + FAISS persistence ============================

def create_embeddings(chunks: List[str], model_name: str) -> List[List[float]]:
    """
    Create one embedding per chunk via Ollama.
    Why: RAG requires vectorisation of chunks for similarity search.
    How: call local embedding model; collect 'embedding' vectors per chunk.
    Assumptions: embedding model is available in Ollama.
    """
    vectors: List[List[float]] = []
    i = 0
    while i < len(chunks):
        resp = ollama.embeddings(model=model_name, prompt=chunks[i])
        vectors.append(resp["embedding"])
        i += 1
    return vectors

def l2_normalise(matrix: List[List[float]]) -> List[List[float]]:
    """
    L2-normalise a matrix of vectors.
    Why: FAISS IndexFlatL2 behaves best with comparable magnitudes.
    How: divide each vector by its Euclidean norm; guard zero norm with 1.0.
    """
    out: List[List[float]] = []
    i = 0
    while i < len(matrix):
        v = matrix[i]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
        i += 1
    return out

def build_or_load_index(index_path: Path):
    """
    Build or load a FAISS index from disk.
    Why: reuse existing index across runs; create new when absent.
    How: read_index if file exists; otherwise return None to init on first add.
    """
    p = Path(index_path)
    if p.exists():
        return faiss.read_index(str(p))
    return None

def add_vectors_to_index(index, matrix: List[List[float]]) -> Tuple[int, int, Any]:
    """
    Add vectors to a FAISS index, creating IndexFlatL2 on first use.
    Why: embedding dimension is known only after first batch; then we can init the index.
    How: convert to float32 array, create IndexFlatL2 if needed, add, and return counts and index.
    """
    xb = np.array(matrix, dtype="float32")
    if index is None:
        dim = xb.shape[1]
        index = faiss.IndexFlatL2(dim)
    previous_total = index.ntotal
    index.add(xb)
    return previous_total, index.ntotal, index

def write_index_to_disk(index, index_path: Path) -> None:
    """
    Persist a FAISS index to disk.
    Why: durability for subsequent queries and restarts.
    How: ensure parent exists; write the index to path.
    """
    p = Path(index_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p))

def append_metadata(metadata_path: Path, records: List[Dict]) -> None:
    """
    Append chunk metadata as JSON Lines.
    Why: keep mapping from vector ids to source chunks for retrieval and audit.
    How: write one compact JSON object per line with vector_id and chunk_text.
    Side effects: appends to file; creates parent folder as needed.
    """
    p = Path(metadata_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        i = 0
        while i < len(records):
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")
            i += 1


# ============================ Sentence-aware chunking (nl/fr/en) ============================

def is_abbreviation(token_lower: str, language: str) -> bool:
    """
    Check if a token is an abbreviation for the language.
    Why: sentence splitting must not break after 'Dhr.' / 'M.' / 'Mr.' etc.
    How: compare lowercased token to language-specific sets.
    """
    return token_lower in ABBREVIATIONS.get(language, set())

def tokenize_sentences(text: str, language: str) -> List[str]:
    """
    Split text into sentences conservatively for nl/fr/en.
    Why: sentence-aligned chunks improve embedding coherence and retrieval quality.
    How: detect ., !, ?, … followed by whitespace + capital; avoid abbreviations; keep delimiter.
    """
    if not text:
        return []
    i, n = 0, len(text)
    out: List[str] = []
    start = 0

    def is_boundary(idx: int) -> bool:
        ch = text[idx]
        if ch not in ".!?…":
            return False
        # Look back a token to avoid abbreviations
        j = idx - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        k = j
        while k >= 0 and not text[k].isspace():
            k -= 1
        token = text[k + 1:idx + 1].strip().lower()
        if is_abbreviation(token, language):
            return False
        # Look ahead past closing quotes and whitespace for capital
        m = idx + 1
        while m < n and text[m] in QUOTE_CLOSE:
            m += 1
        while m < n and text[m].isspace():
            m += 1
        if m < n and text[m].isupper():
            return True
        return False

    while i < n:
        if text[i] in ".!?…" and is_boundary(i):
            end = i + 1
            out.append(text[start:end].strip())
            j = end
            while j < n and text[j].isspace():
                j += 1
            start = j
            i = j
            continue
        i += 1

    if start < n:
        tail = text[start:].strip()
        if tail:
            out.append(tail)
    return out

def create_overlapping_chunks(text: str, size: int, overlap: int) -> List[str]:
    """
    Create overlapping character-based chunks (fallback).
    Why: robust default when sentence tokenisation fails (e.g., noisy OCR).
    How: slide a window of 'size' with 'overlap' characters between windows.
    """
    if size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        start = start + size - overlap
    return chunks

def create_sentence_chunks(text: str, language: str, max_chars: int, approx_overlap_chars: int) -> List[str]:
    """
    Create chunks that respect sentence boundaries with approximate overlap.
    Why: chunking on sentence boundaries reduces mid-sentence cuts and improves RAG quality.
    How: pack sentences up to max_chars; compute sentence overlap from average sentence length.
    """
    if max_chars <= 0:
        raise ValueError("chunk_size must be > 0")
    if approx_overlap_chars < 0 or approx_overlap_chars >= max_chars:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    sents = tokenize_sentences(text, language)
    if not sents:
        return create_overlapping_chunks(text, max_chars, approx_overlap_chars)

    avg_len = max(1, int(sum(len(s) for s in sents) / len(sents)))
    sent_overlap = max(0, min(len(sents) - 1, approx_overlap_chars // avg_len))

    chunks: List[str] = []
    i = 0
    while i < len(sents):
        buf: List[str] = []
        length = 0
        j = i
        while j < len(sents):
            s = sents[j]
            s_len = len(s) + (1 if length > 0 else 0)
            if length + s_len > max_chars and length > 0:
                break
            buf.append(s)
            length += s_len
            j += 1
        if not buf:
            buf = [sents[j]]
            j += 1
        chunks.append(" ".join(buf))
        if j >= len(sents):
            break
        i = max(i + len(buf) - sent_overlap, j) if sent_overlap > 0 else j

    return chunks


# ============================ Ingest (RAG) pipeline ============================

def hash_path(path: Path) -> str:
    """
    Produce a stable, short document id from an absolute path.
    Why: keep chunks grouped by source and reproducible across runs.
    How: SHA-256 of resolved path, truncated for readability.
    """
    hasher = hashlib.sha256()
    hasher.update(str(Path(path).resolve()).encode("utf-8"))
    return hasher.hexdigest()[:16]

def process_text_to_faiss(
    original_text: str,
    language: str,
    source_path: Path,
    index_path: Path,
    metadata_path: Path,
    redacted_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    llm_pii_model_name: str,
) -> int:
    """
    Anonymise text and persist embeddings + metadata.
    Why: ingestion must store only filtered content for privacy-preserving RAG.
    How: filter_all → sentence chunks → embeddings → FAISS write → append JSONL metadata.
    Side effects: writes index_path and metadata_path; writes redacted sidecar to redacted_path.
    """
    filtered = filter_all(
        original_text=original_text,
        language=language,
        llm_pii_model_name=llm_pii_model_name,
        debug_report_path=Path("data") / "redaction_report.json",
    )

    # Write redacted sidecar near runtime artifacts
    if redacted_path is not None:
        redacted_path.mkdir(parents=True, exist_ok=True)
        base = Path(source_path).name or "document"
        out_name = f"{Path(base).stem}.redacted.txt"
        out_path = redacted_path / out_name
        out_path.write_text(filtered, encoding="utf-8")

    # Sentence-aware chunking
    chunks = create_sentence_chunks(filtered, language, chunk_size, chunk_overlap)

    # Embeddings
    matrix = create_embeddings(chunks, model_name=embedding_model)
    matrix = l2_normalise(matrix)

    # FAISS
    index = build_or_load_index(index_path)
    previous_total, new_total, index = add_vectors_to_index(index, matrix)
    write_index_to_disk(index, index_path)

    # Metadata JSONL
    records: List[Dict] = []
    doc_id = hash_path(source_path)
    i = 0
    while i < len(chunks):
        vector_id = previous_total + i
        records.append({
            "vector_id": vector_id,
            "document_id": doc_id,
            "source_path": str(source_path),
            "chunk_index": i,
            "chunk_text": chunks[i],  # already filtered text
        })
        i += 1
    append_metadata(metadata_path, records)

    return new_total - previous_total


# ============================ Self-test ============================

if __name__ == "__main__":
    """
    Minimal self-test to sanity-check anonymisation without the server.
    Why: instant feedback that anchors are preserved and placeholders are clean.
    How: run a few multilingual samples and print input/output pairs.
    Assumptions: Ollama is running with the specified models locally.
    """
    samples = [
        ("De heer SCHUPP Carlo, geboren te Brussel op 2 mei 1980.", "nl"),
        ("Monsieur Jean Dupont, né à Paris le 14 juillet 1975.", "fr"),
        ("Mr. John Smith, born in London on March 5, 1970.", "en"),
        ("Mevrouw VANMARCKE Kristina, wonende te Gent, IBAN BE12 3456 7890 1234.", "nl"),
    ]
    print("=== Anonymiser self-test ===")
    for txt, lang in samples:
        print("\n--- Input ---")
        print(txt)
        try:
            out = filter_all(
                original_text=txt,
                language=lang,
                llm_pii_model_name="mistral",
            )
            print("--- Output ---")
            print(out)
        except Exception as exc:
            print(f"Error: {exc}")
    print("\n=== End of self-test ===")
