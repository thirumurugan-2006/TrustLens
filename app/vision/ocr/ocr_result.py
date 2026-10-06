"""
TrustLens Step 11 — Common OCR Result Schema
=============================================

Separates raw_confidence (what the OCR engine returns) from
reliability_score (TrustLens-calibrated estimate of correctness).

Never overwrite raw_confidence with a derived value.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# OCR Status values
# ---------------------------------------------------------------------------
OCR_STATUS_SUCCESS = "SUCCESS"
OCR_STATUS_PARTIAL = "PARTIAL"
OCR_STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"
OCR_STATUS_LANGUAGE_UNCERTAIN = "LANGUAGE_UNCERTAIN"
OCR_STATUS_FAILED = "OCR_FAILED"


@dataclass
class TamilOCRResult:
    """
    Unified OCR result for a single image used throughout TrustLens.

    Fields
    ------
    text : str
        OCR output text (joined from all detected regions).
    language : str
        Language code detected or assigned: "ta", "en", "mixed", "unknown".
    raw_confidence : float | None
        The value directly returned by EasyOCR (average across regions).
        This is never modified or overwritten.
    reliability_score : float | None
        TrustLens-calibrated estimate of OCR correctness (0.0–1.0).
        Computed from multiple signals; may differ substantially from
        raw_confidence.
    bbox : list | None
        Bounding boxes for each detected text region.
    backend : str
        Name of the OCR backend that produced this result.
    status : str
        One of SUCCESS, PARTIAL, LOW_CONFIDENCE, LANGUAGE_UNCERTAIN, OCR_FAILED.
    tamil_ratio : float
        Fraction of characters in the Tamil Unicode block (U+0B80–U+0BFF).
    valid_char_ratio : float
        Fraction of characters that are recognised Tamil/Latin/numeric characters.
    detection_count : int
        Number of text regions detected.
    region_confidences : list[float]
        Per-region confidence values from the OCR engine.
    reliable : bool
        True if reliability_score is above the operational threshold.
    error : str
        Error message if OCR failed.
    processing_time_ms : float
        Wall-clock time for OCR in milliseconds.
    """

    text: str = ""
    raw_text: str = ""
    normalized_text: str = ""
    language: str = "unknown"
    raw_confidence: Optional[float] = None
    reliability_score: Optional[float] = None
    bbox: Optional[list] = None
    backend: str = "easyocr"
    status: str = OCR_STATUS_FAILED
    tamil_ratio: float = 0.0
    valid_char_ratio: float = 0.0
    detection_count: int = 0
    region_confidences: list = field(default_factory=list)
    reliable: bool = False
    error: str = ""
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkEntry:
    """
    A single row in the benchmark CSV linking image, ground truth, and OCR result.
    """
    image_id: str
    ground_truth: Optional[str]
    prediction: str
    raw_confidence: Optional[float]
    reliability_score: Optional[float]
    category: str
    language: str
    backend: str
    status: str
    processing_time_ms: float
    # Metrics (None when ground truth is missing)
    cer: Optional[float] = None
    wer: Optional[float] = None
    exact_match: Optional[int] = None
    annotation_status: str = "MISSING"
    confidence_bucket: str = "unavailable"
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
