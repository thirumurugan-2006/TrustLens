import unicodedata
import re

def normalize_unicode(text: str) -> str:
    """Safely normalizes Unicode to NFC form."""
    if not text:
        return ""
    # NFC is generally safe and preferred for Tamil
    return unicodedata.normalize("NFC", text)

def remove_control_characters(text: str) -> str:
    """Removes OCR-generated control characters while keeping safe whitespace."""
    if not text:
        return ""
    # Remove zero-width spaces and control characters (C0/C1)
    # except for newline and tab
    text = re.sub(r'[\u200B-\u200F\u202A-\u202E\uFEFF]', '', text)
    # Remove control characters except \n, \r, \t
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ('\n', '\r', '\t'))
    return text

def normalize_whitespace(text: str) -> str:
    """Normalizes whitespace while preserving structural line breaks."""
    if not text:
        return ""
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Clean up spaces around newlines first
    text = re.sub(r' \n', '\n', text)
    text = re.sub(r'\n ', '\n', text)
    # Replace multiple newlines with a double newline to preserve paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def normalize_punctuation(text: str) -> str:
    """Normalizes basic punctuation without destroying formatting."""
    if not text:
        return ""
    # Normalize quotes
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('—', '-')
    return text
