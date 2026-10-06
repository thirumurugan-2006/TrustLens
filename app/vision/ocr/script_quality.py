"""
TrustLens OCR v2 — Script Quality Utilities
=============================================

Provides Tamil Unicode detection, Latin detection, and script distribution
calculations.  Used by the OCR quality evaluator and candidate selector.

Tamil Unicode block: U+0B80–U+0BFF
Latin script range used: U+0041–U+024F  (Basic Latin + Latin Extended)

Important rules:
  - Only alphabetic characters (Unicode category "L*") are counted.
  - Digits, punctuation, and currency symbols are EXCLUDED from all ratios.
  - "₹500" contains zero letter-class characters → does NOT inflate any ratio.
  - Numbers alone cannot classify a screenshot as English.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Unicode ranges
# ---------------------------------------------------------------------------
TAMIL_START = 0x0B80
TAMIL_END   = 0x0BFF

LATIN_START = 0x0041
LATIN_END   = 0x024F   # Latin + Latin Extended-A/B

# Valid characters for the valid_char_ratio check
_VALID_CHAR_RE = re.compile(
    r"[\u0B80-\u0BFF"               # Tamil block
    r"\u0020"                        # space
    r"0-9"                           # digits
    r"A-Za-z"                        # ASCII letters
    r"\.,!?\-\(\)\[\]\"\u2018\u2019\u201c\u201d\u2026"  # punctuation
    r"\u20B9\u0024\u20AC\u00A3"     # currency symbols (₹ $ € £)
    r"]"
)


# ---------------------------------------------------------------------------
# Core character tests
# ---------------------------------------------------------------------------

def _is_letter(ch: str) -> bool:
    """Return True if the character's Unicode category starts with 'L'."""
    return unicodedata.category(ch).startswith("L")


def contains_tamil(text: str) -> bool:
    """Return True if any Tamil Unicode character is present."""
    return any(TAMIL_START <= ord(ch) <= TAMIL_END for ch in text)


def tamil_char_count(text: str) -> int:
    """Count Tamil Unicode characters (letter-class only)."""
    return sum(1 for ch in text if _is_letter(ch) and TAMIL_START <= ord(ch) <= TAMIL_END)


def latin_char_count(text: str) -> int:
    """Count Latin-script characters (letter-class only)."""
    return sum(1 for ch in text if _is_letter(ch) and LATIN_START <= ord(ch) <= LATIN_END)


# ---------------------------------------------------------------------------
# Ratio functions
# ---------------------------------------------------------------------------

def tamil_ratio(text: str) -> float:
    """
    Fraction of letter-class characters that are in the Tamil Unicode block.

    Returns 0.0 for empty / whitespace-only text or when no letters exist.
    Numbers and punctuation do NOT affect this ratio.
    """
    letters = [ch for ch in text if _is_letter(ch)]
    if not letters:
        return 0.0
    ta_count = sum(1 for ch in letters if TAMIL_START <= ord(ch) <= TAMIL_END)
    return ta_count / len(letters)


def latin_ratio(text: str) -> float:
    """
    Fraction of letter-class characters that are Latin-script.

    Returns 0.0 for empty / whitespace-only text or when no letters exist.
    Numbers and punctuation do NOT affect this ratio.
    """
    letters = [ch for ch in text if _is_letter(ch)]
    if not letters:
        return 0.0
    la_count = sum(1 for ch in letters if LATIN_START <= ord(ch) <= LATIN_END)
    return la_count / len(letters)


def script_distribution(text: str) -> dict[str, float]:
    """
    Fraction of letter-class characters in each script group.

    Returns
    -------
    dict with keys "Tamil", "Latin", "Other".
    Values sum to 1.0 (or all 0.0 when no letters present).
    """
    counts: dict[str, int] = {"Tamil": 0, "Latin": 0, "Other": 0}
    for ch in text:
        if not _is_letter(ch):
            continue
        cp = ord(ch)
        if TAMIL_START <= cp <= TAMIL_END:
            counts["Tamil"] += 1
        elif LATIN_START <= cp <= LATIN_END:
            counts["Latin"] += 1
        else:
            counts["Other"] += 1
    total = sum(counts.values())
    if total == 0:
        return {"Tamil": 0.0, "Latin": 0.0, "Other": 0.0}
    return {k: round(v / total, 4) for k, v in counts.items()}


def valid_char_ratio(text: str) -> float:
    """
    Fraction of non-whitespace characters that are recognised Tamil/Latin/digit/
    punctuation/currency characters.

    A very low ratio indicates junk/garbage OCR output.
    Returns 0.0 for empty text.
    """
    stripped = text.replace(" ", "").replace("\n", "").replace("\r", "")
    if not stripped:
        return 0.0
    matches = _VALID_CHAR_RE.findall(stripped)
    return len(matches) / len(stripped)


def alphanumeric_ratio(text: str) -> float:
    """
    Fraction of non-whitespace characters that are alphanumeric (letters or digits).

    Returns 0.0 for empty text.
    """
    non_ws = [ch for ch in text if not ch.isspace()]
    if not non_ws:
        return 0.0
    alphanum = sum(1 for ch in non_ws if ch.isalnum())
    return alphanum / len(non_ws)


def script_ratios(text: str) -> tuple[float, float]:
    """Return (tamil_ratio, latin_ratio) as a convenience tuple."""
    dist = script_distribution(text)
    return dist["Tamil"], dist["Latin"]


def detect_language_label(text: str, raw_confidence: float | None = None) -> str:
    """
    Determine a language label from script evidence alone.

    This function NEVER assigns language="en" because Tamil confidence is low.
    Uncertainty is always reported explicitly.

    Decision tree:
      1. If Tamil fraction >= 0.5  → "ta"
      2. If both Tamil and Latin present (>0) → "mixed"
      3. If Latin fraction >= 0.5  → "en"
      4. If some Tamil (>0)        → "ta"
      5. If some Latin (>0)        → "en"
      6. Otherwise                 → "unknown"
    """
    dist = script_distribution(text)
    ta_f = dist["Tamil"]
    la_f = dist["Latin"]

    if ta_f >= 0.5:
        return "ta"
    if ta_f > 0.0 and la_f > 0.0:
        return "mixed"
    if la_f >= 0.5:
        return "en"
    if ta_f > 0.0:
        return "ta"
    if la_f > 0.0:
        return "en"
    return "unknown"
