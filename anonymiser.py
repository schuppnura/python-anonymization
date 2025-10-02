import re
from constants import PII_REGEX_PATTERNS

def apply_regex_redaction(text: str) -> str:
    redacted = text
    for pii_type, pattern in PII_REGEX_PATTERNS.items():
        redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
    return redacted

def decide_sentence_is_sensitive(sentence: str) -> bool:
    # Placeholder: integrate local LLM call later
    return False

def apply_contextual_redaction(text: str, use_llm_privacy_filter: bool) -> str:
    if not use_llm_privacy_filter:
        return text
    sentences = split_sentences_simple(text)
    safe_sentences: list[str] = []
    for s in sentences:
        if decide_sentence_is_sensitive(s):
            safe_sentences.append("[REDACTED_CONTEXT]")
        else:
            safe_sentences.append(s)
    return " ".join(safe_sentences)

def split_sentences_simple(text: str) -> list[str]:
    candidates = [t.strip() for t in re.split(r"(?<=[.!?])\s+", text)]
    return [c for c in candidates if c]
