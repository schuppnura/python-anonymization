from anonymiser import apply_regex_redaction

def test_apply_regex_redaction_email():
    text = "Contact me at user@example.com for details."
    redacted = apply_regex_redaction(text)
    assert "user@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
