# Constants shared across modules
EMBEDDING_DIMENSION: int = 1024
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".pdf", ".docx")
PII_REGEX_PATTERNS: dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"\b(?:\+?\d[\d\s\-]{6,}\d)\b",
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b"
}
