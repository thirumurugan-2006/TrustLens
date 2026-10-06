"""
TrustLens — Tamil Script Utilities (OCR v2 compatible)
=======================================================

Provides Tamil Unicode detection, character ratio calculations, and
language label detection.

This module now delegates to app.vision.ocr.script_quality for all
ratio and distribution computations so there is a single source of truth.

The functions below are kept for backwards compatibility with existing
callers (combined_ocr.py, tamil_reliability.py, etc.).

Do NOT use Tamil Unicode detection as the only reliability signal.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export from script_quality (single source of truth)
# ---------------------------------------------------------------------------

from app.vision.ocr.script_quality import (
    TAMIL_START as TAMIL_BLOCK_START,
    TAMIL_END   as TAMIL_BLOCK_END,
    contains_tamil,
    tamil_char_count,
    tamil_ratio,
    latin_ratio,
    valid_char_ratio,
    script_distribution,
    detect_language_label,
)

__all__ = [
    "TAMIL_BLOCK_START",
    "TAMIL_BLOCK_END",
    "contains_tamil",
    "tamil_char_count",
    "tamil_ratio",
    "latin_ratio",
    "valid_char_ratio",
    "script_distribution",
    "detect_language_label",
]
