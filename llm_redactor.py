# llm_redactor.py
# Purpose: Use a local LLM (via Ollama) to interpret PERSON names in arbitrary text,
# map them back to character spans in the original text, and mask them BEFORE
# any indexing. Designed to be privacy-first: text never leaves the machine.
#
# Assumptions:
# - Ollama is installed and the chosen model is available locally.
# - The calling pipeline handles I/O; this module returns redacted text only.
#
# Side effects:
# - Optional: writes a JSON redaction report if 'write_report_to' is provided.

from __future__ import annotations

from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import unicodedata
import string

try:
    import ollama  # local inference host
except Exception:
    ollama = None  # degrade gracefully if not present


SYSTEM_PROMPT_EXTRACT: str = (
    "You are a named-entity recognizer. Identify all PERSON names.\n"
    "Return STRICT JSON only with this schema:\n"
    "{ \"persons\": [ {\"given\": [\"...\"], \"family\": [\"...\"]}, ... ] }\n"
    "- Put all given names (in order) in 'given'.\n"
    "- Put all family-name tokens in 'family'.\n"
    "- EXCLUDE titles, roles, and legal qualifiers (e.g., 'De heer', 'Mevrouw', 'voornoemd').\n"
    "- Do not include organizations, locations, or dates."
)


def call_ollama_json(model_name: str, system: str, user: str) -> Dict[str, Any]:
    """
    Call a local LLM via Ollama expecting strict JSON output.
    Assumptions: model is installed; Ollama is running.
    Side effects: local model inference.
    Failure mode: returns {} if anything goes wrong (caller should handle).
    """
    if ollama is None:
        return {}
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": 0.0, "format": "json"},
        )
        content = response["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Best-effort salvage of a JSON object if model is slightly off-format.
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    return {}
            return {}
    except Exception:
        return {}


def extract_person_structs_llm(text: str, model_name: str) -> List[Dict[str, List[str]]]:
    """
    Ask the LLM to return PERSON names as structured tokens: given[] and family[].
    Assumptions: JSON schema enforced by prompt; still validate defensively.
    Side effects: local inference.
    """
    user_prompt = "Extract PERSON names as {persons:[{given:[],family:[]}]}.\nText:\n" + text
    obj = call_ollama_json(model_name=model_name, system=SYSTEM_PROMPT_EXTRACT, user=user_prompt)
    persons = obj.get("persons", []) if isinstance(obj, dict) else []
    cleaned: List[Dict[str, List[str]]] = []
    for entry in persons:
        if isinstance(entry, dict):
            given_raw = entry.get("given", [])
            family_raw = entry.get("family", [])
            given = [t.strip() for t in given_raw if isinstance(t, str) and t.strip()]
            family = [t.strip() for t in family_raw if isinstance(t, str) and t.strip()]
            if given or family:
                cleaned.append({"given": given, "family": family})
    return cleaned


def create_variants_for_person(person: Dict[str, List[str]]) -> List[str]:
    """
    Generate search candidates from tokens so we catch both surface forms:
      - 'given given ... family' and 'family given given ...'
    Assumptions: person dict contains 'given' and 'family' lists of tokens.
    Side effects: none.
    """
    given = " ".join(person.get("given", []))
    family = " ".join(person.get("family", []))
    variants: List[str] = []
    if given and family:
        variants.append(f"{given} {family}".strip())
        variants.append(f"{family} {given}".strip())
    elif family:
        variants.append(family)
    elif given:
        variants.append(given)
    # De-duplicate preserving order
    deduped: List[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(variants):
        v = variants[i]
        if v not in seen:
            deduped.append(v)
            seen.add(v)
        i += 1
    return deduped


def normalise_for_compare(s: str) -> str:
    """
    Normalise strings for tolerant matching:
    - strip accents, casefold, remove punctuation, collapse whitespace.
    Assumptions: ASCII fallbacks acceptable for matching.
    Side effects: none.
    """
    s_norm = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s_norm = s_norm.casefold()
    table = str.maketrans("", "", string.punctuation)
    s_norm = s_norm.translate(table)
    s_norm = " ".join(s_norm.split())
    return s_norm


def build_index_map(original: str) -> Tuple[str, List[int]]:
    """
    Build a normalised view of the original text and an index map back to original
    character positions. We drop punctuation and compress sequences of whitespace.
    Assumptions: simple heuristic map is sufficient for masking.
    Side effects: none.
    """
    norm_chars: List[str] = []
    idx_map: List[int] = []
    last_was_space = False
    i = 0
    n = len(original)
    while i < n:
        ch = original[i]
        decomp = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
        if decomp:
            if decomp in string.punctuation:
                # skip punctuation but do not advance map
                pass
            else:
                if decomp.isspace():
                    if not last_was_space:
                        norm_chars.append(" ")
                        idx_map.append(i)
                        last_was_space = True
                else:
                    norm_chars.append(decomp.casefold())
                    idx_map.append(i)
                    last_was_space = False
        i += 1
    norm = "".join(norm_chars).strip()
    return norm, idx_map


def find_spans_flexible(text: str, needle: str) -> List[Tuple[int, int]]:
    """
    Find all [start, end) spans for 'needle' in 'text' with tolerant matching:
    - case-insensitive
    - punctuation-insensitive
    - whitespace-insensitive
    Assumptions: suitable for legal texts with varied formatting.
    Side effects: none.
    """
    spans: List[Tuple[int, int]] = []
    if not needle or not needle.strip():
        return spans
    norm_hay, map_hay = build_index_map(text)
    norm_needle = normalise_for_compare(needle)

    start = 0
    hay_len = len(norm_hay)
    needle_len = len(norm_needle)
    if needle_len == 0:
        return spans

    # Loop without 'break': exit when idx == -1 by toggling a done flag
    done = False
    while not done:
        idx = norm_hay.find(norm_needle, start)
        if idx == -1:
            done = True
        else:
            start_orig = map_hay[idx]
            end_norm_pos = idx + needle_len - 1
            if end_norm_pos < len(map_hay):
                end_orig = map_hay[end_norm_pos] + 1
            else:
                end_orig = len(text)
            spans.append((start_orig, end_orig))
            start = idx + needle_len
            if start >= hay_len:
                done = True
    return spans


def merge_overlapping_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping or adjacent spans to avoid double-masking artifacts.
    Assumptions: small number of spans per block; O(n log n) sort acceptable.
    Side effects: none.
    """
    if not spans:
        return spans
    spans_sorted = sorted(spans, key=lambda p: (p[0], p[1]))
    merged: List[Tuple[int, int]] = []
    i = 0
    m = len(spans_sorted)
    while i < m:
        s, e = spans_sorted[i]
        if not merged:
            merged.append((s, e))
        else:
            ls, le = merged[-1]
            if s <= le:
                merged[-1] = (ls, e if e > le else le)
            else:
                merged.append((s, e))
        i += 1
    return merged


def apply_mask(text: str, spans: List[Tuple[int, int]], placeholder: str) -> str:
    """
    Apply masking from right to left so earlier indices remain valid.
    Assumptions: spans are [start, end) in original text coordinates.
    Side effects: information is irreversibly removed in the returned value.
    """
    result = text
    i = len(spans) - 1
    while i >= 0:
        s, e = spans[i]
        if e > s:
            result = result[:s] + placeholder + result[e:]
        i -= 1
    return result


def split_into_blocks(text: str, max_chars: int) -> List[str]:
    """
    Split text into blocks of size <= max_chars to help the LLM recall entities
    in long inputs. Assumptions: simple contiguous split is sufficient for MVP.
    Side effects: none.
    """
    if len(text) <= max_chars:
        return [text]
    blocks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = start + max_chars
        if end > n:
            end = n
        blocks.append(text[start:end])
        start = end
    return blocks


def redact_block(text: str, model_name: str, placeholder: str) -> Tuple[str, Dict[str, Any]]:
    """
    Redact a single block: interpret PERSONs via LLM → generate variants → map spans → mask.
    Returns redacted block and a small report for diagnostics.
    Assumptions: model is local; tolerant mapping suffices for this block.
    Side effects: local inference.
    """
    persons = extract_person_structs_llm(text, model_name)
    variants: List[str] = []
    spans: List[Tuple[int, int]] = []

    # Generate both orderings per person and collect spans
    j = 0
    total = len(persons)
    while j < total:
        for cand in create_variants_for_person(persons[j]):
            variants.append(cand)
            spans.extend(find_spans_flexible(text, cand))
        j += 1

    merged_spans = merge_overlapping_spans(spans)
    redacted = apply_mask(text, merged_spans, placeholder)

    # Simple string-based post-check: which variants still present?
    remaining: List[str] = []
    k = 0
    vlen = len(variants)
    while k < vlen:
        cand = variants[k]
        leftover = find_spans_flexible(redacted, cand)
        if len(leftover) > 0:
            remaining.append(cand)
        k += 1

    report = {
        "persons_struct": persons,
        "detected_variants": variants,
        "spans": merged_spans,
        "remaining_variants_after_mask": remaining,
    }
    return redacted, report


def redact_person_names_using_llm(
    text: str,
    model_name: str,
    max_block_chars: int = 2000,
    placeholder: str = "[REDACTED_PERSON]",
    write_report_to: Path | None = None,
) -> str:
    """
    Top-level API expected by the pipeline/anonymiser.
    - Splits text into blocks for better LLM recall.
    - For each block, interprets PERSON names and masks them with tolerant matching.
    - Optionally writes a JSON report for QA.
    Assumptions: called BEFORE chunking/embedding; caller handles I/O.
    Side effects: local inference; optional file write for report.
    """
    if ollama is None:
        # Ollama not available: return text unchanged to avoid breaking ingestion.
        return text

    blocks = split_into_blocks(text, max_block_chars)
    redacted_blocks: List[str] = []
    reports: List[Dict[str, Any]] = []

    i = 0
    bcount = len(blocks)
    while i < bcount:
        red_b, rep = redact_block(blocks[i], model_name, placeholder)
        redacted_blocks.append(red_b)
        reports.append(rep)
        i += 1

    redacted_text = "".join(redacted_blocks)

    if write_report_to is not None:
        write_report_to.parent.mkdir(parents=True, exist_ok=True)
        report_obj = {"blocks": reports}
        write_report_to.write_text(json.dumps(report_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return redacted_text
