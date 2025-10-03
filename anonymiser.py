# anonymiser.py
import re
from pathlib import Path
from constants import PII_REGEX_PATTERNS
from llm_redactor import redact_block, redact_person_names_using_llm

def apply_regex_redaction(text: str) -> str:
    redacted = text
    for pii_type, pattern in PII_REGEX_PATTERNS.items():
        redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
    return redacted

def apply_person_redaction_with_llm(text: str, model_name: str, enabled: bool, debug_report_path: Path | None = None) -> str:
    if not enabled:
        return text
    # Write one redaction report per ingest to inspect detections
    return redact_person_names_using_llm(
        text=text,
        model_name=model_name,
        max_block_chars=2000,
        placeholder="[REDACTED_PERSON]",
        write_report_to=debug_report_path,
    )
