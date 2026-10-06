"""
TrustLens OCR v2 — Pydantic Schemas
=====================================

Structured schemas for OCR candidates and the final selected result.
All fields are documented. Schema is extensible via metadata dict.

OCRCandidate
    Represents one (preprocessing × language_mode) OCR attempt.

OCRResult
    The selected best candidate plus debug information.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# OCR Candidate — one (preprocessing × language_mode) attempt
# ---------------------------------------------------------------------------

class OCRCandidate(BaseModel):
    """
    A single OCR attempt produced by running one language reader
    on one preprocessing variant of an image.

    Fields
    ------
    language_mode : list[str]
        EasyOCR / PaddleOCR language codes used (e.g., ["en"] or ["ta"]).
    preprocessing : str
        Name of the preprocessing variant applied before OCR
        (e.g., "original", "upscale_2x", "grayscale_upscale_2x",
        "contrast_enhanced", "adaptive_threshold").
    confidence : float
        Raw average confidence returned by the OCR engine across all
        detected text regions.  Never modified or overwritten.
    detection_count : int
        Number of text regions detected by the OCR engine.
    text_length : int
        Character count of the joined OCR output text.
    tamil_ratio : float
        Fraction of letter-class characters (Unicode category "L*") that
        fall in the Tamil Unicode block U+0B80–U+0BFF.
        0.0 = no Tamil letters, 1.0 = all letters are Tamil.
    latin_ratio : float
        Fraction of letter-class characters in the Latin script range
        U+0041–U+024F.
        Numbers and punctuation are excluded so "₹500" does not inflate this.
    script_distribution : dict[str, float]
        {"Tamil": float, "Latin": float, "Other": float}
    alphanumeric_ratio : float
        Fraction of non-whitespace characters that are alphanumeric.
        Very low values indicate junk/garbage OCR output.
    valid_char_ratio : float
        Fraction of characters that match a known Tamil/Latin/numeric pattern.
    selection_score : float
        Composite quality score used for candidate ranking.
        Formula (documented in ocr_quality.py):
            0.45 * confidence
          + 0.25 * script_quality
          + 0.15 * text_quality
          + 0.15 * detection_quality
    usable : bool
        True only when confidence >= 0.60 AND the text is non-empty.
        A candidate must be usable before it can be selected as the winner.
    warnings : list[str]
        Human-readable quality warnings for this candidate.
    text : str
        The OCR text produced for this candidate.  Preserved for debugging.
    backend : str
        Name of the OCR backend that produced this result
        (e.g., "easyocr_en", "paddleocr_ta").
    """

    language_mode: List[str] = Field(default_factory=list)
    preprocessing: str = "original"
    confidence: float = 0.0
    detection_count: int = 0
    text_length: int = 0
    tamil_ratio: float = 0.0
    latin_ratio: float = 0.0
    script_distribution: Dict[str, float] = Field(default_factory=dict)
    alphanumeric_ratio: float = 0.0
    valid_char_ratio: float = 0.0
    selection_score: float = 0.0
    usable: bool = False
    warnings: List[str] = Field(default_factory=list)
    text: str = ""
    backend: str = "unknown"

    # Allow future extension without schema changes
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# OCR Result — the selected best candidate + debug info
# ---------------------------------------------------------------------------

class OCRResult(BaseModel):
    """
    The final OCR result after candidate selection.

    The selected candidate's fields are promoted to the top level.
    All candidates are preserved in ``candidates`` for debugging.

    Raw OCR text is always preserved in ``raw_text``.
    The post-normalized text is stored in ``normalized_text``.
    ``text`` is the primary text alias (set to normalized_text when available).

    Fields
    ------
    text : str
        Primary text — the normalized OCR output of the selected candidate.
    raw_text : str
        Raw OCR text exactly as returned by the engine.  Never overwritten.
    normalized_text : str
        Unicode-normalized, whitespace-collapsed version of raw_text.
    confidence : float
        Raw confidence of the selected candidate (engine value, never inflated).
    language_mode : list[str]
        Language codes used by the winning candidate.
    preprocessing : str
        Preprocessing variant that produced the winning candidate.
    backend : str
        OCR backend name of the winning candidate.
    detection_count : int
        Number of text regions in the winning candidate.
    text_length : int
        Character count of the winning candidate text.
    script_distribution : dict[str, float]
        Script breakdown of the winning candidate text.
    tamil_ratio : float
        Tamil character ratio of the winning candidate text.
    latin_ratio : float
        Latin character ratio of the winning candidate text.
    selection_score : float
        Composite quality score of the winning candidate.
    usable : bool
        True if the winning candidate passed the hard reliability gate.
    unreliable : bool
        True if NO usable candidate existed and the fallback was used.
        When True the language_analysis should return "unknown".
    warnings : list[str]
        Aggregated quality warnings.
    candidates : list[OCRCandidate]
        All OCR candidates evaluated.  Preserved for debugging and research.
    attempts : int
        Total number of OCR attempts made (preprocessing × language_mode).
    ocr_engine : str
        Canonical engine identifier string.
    ocr_version : str
        Version of the OCR library used.
    """

    text: str = ""
    raw_text: str = ""
    normalized_text: str = ""
    confidence: float = 0.0
    language_mode: List[str] = Field(default_factory=list)
    preprocessing: str = "original"
    backend: str = "unknown"
    detection_count: int = 0
    text_length: int = 0
    script_distribution: Dict[str, float] = Field(default_factory=dict)
    tamil_ratio: float = 0.0
    latin_ratio: float = 0.0
    selection_score: float = 0.0
    usable: bool = False
    unreliable: bool = True
    warnings: List[str] = Field(default_factory=list)
    candidates: List[OCRCandidate] = Field(default_factory=list)
    attempts: int = 0
    ocr_engine: str = "easyocr_en+paddleocr_ta"
    ocr_version: str = ""

    # Allow future extension
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)
