"""
TrustLens OCR v2 — OCR Quality Evaluator
==========================================

Evaluates a single OCR candidate and returns quality metrics.

Hard reliability gate
---------------------
A candidate with confidence < MIN_OCR_CONFIDENCE (0.60) is ALWAYS marked
usable=False.  No selection_score can override this gate.

Selection score formula (documented)
-------------------------------------
selection_score =
    0.45 * normalized_confidence   (primary engine signal)
  + 0.25 * script_quality          (Tamil or Latin match for expected language)
  + 0.15 * text_quality            (normalised text length)
  + 0.15 * detection_quality       (normalised detection count)

All components are in [0.0, 1.0].

Corruption / garbage detection signals (informational — never delete text)
--------------------------------------------------------------------------
- very low alphanumeric ratio         → suspicious
- very low valid_char_ratio           → suspicious
- unexpected script for language mode → suspicious
- extremely short text vs many detections → suspicious
"""
from __future__ import annotations

from app.vision.ocr.script_quality import (
    alphanumeric_ratio as _alphanumeric_ratio,
    valid_char_ratio as _valid_char_ratio,
    script_distribution,
    tamil_ratio as _tamil_ratio,
    latin_ratio as _latin_ratio,
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Hard gate: below this confidence a candidate is NEVER usable
MIN_OCR_CONFIDENCE: float = 0.60

# Text quality normalisation target
_TEXT_LEN_TARGET = 50    # characters for full text_quality score
_DETECTION_TARGET = 10   # detections for full detection_quality score


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def is_ocr_usable(result: dict) -> bool:
    """
    Hard reliability gate.

    A candidate is usable only when ALL conditions pass:
      1. Non-empty text
      2. confidence >= MIN_OCR_CONFIDENCE (0.60)
      3. For Tamil mode: at least some Tamil characters present

    Important: a high selection_score cannot override this gate.
    Example:
        confidence = 0.265, selection_score = 0.63 → usable = False
    """
    text = result.get("text", "").strip()
    if not text:
        return False

    conf = float(result.get("confidence", 0.0))
    if conf < MIN_OCR_CONFIDENCE:
        return False

    mode = result.get("language_mode", [])
    if mode == ["ta"]:
        dist = result.get("script_distribution") or script_distribution(text)
        if dist.get("Tamil", 0.0) <= 0.0:
            return False

    return True


def selection_score(result: dict) -> float:
    """
    Compute a transparent composite quality score for an OCR candidate.

    Formula
    -------
    selection_score =
        0.45 * normalized_confidence   — raw engine confidence
      + 0.25 * script_quality          — how well the script matches the reader
      + 0.15 * text_quality            — normalised text length signal
      + 0.15 * detection_quality       — normalised detection count signal

    Script quality is script-aware:
        Tamil mode → reward high Tamil ratio
        English mode → reward high Latin ratio

    This formula is intentionally transparent.  Adjust weights here only,
    not inside the candidate selector.
    """
    text = result.get("text", "")
    conf = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    mode = result.get("language_mode", [])

    # Use pre-computed distribution if available, else compute on-the-fly
    dist = result.get("script_distribution") or script_distribution(text)

    if mode == ["ta"]:
        script_quality = dist.get("Tamil", 0.0)
    else:
        script_quality = dist.get("Latin", 0.0)

    text_quality = min(len(text) / _TEXT_LEN_TARGET, 1.0)
    detection_quality = min(int(result.get("detection_count", 0)) / _DETECTION_TARGET, 1.0)

    score = (
        0.45 * conf
        + 0.25 * script_quality
        + 0.15 * text_quality
        + 0.15 * detection_quality
    )
    return round(score, 4)


def quality_warnings(result: dict) -> list[str]:
    """
    Generate human-readable quality warnings for an OCR candidate.

    These are informational signals — they never delete or alter text.
    """
    warnings: list[str] = []
    text = result.get("text", "")
    conf = float(result.get("confidence", 0.0))
    mode = result.get("language_mode", [])

    if conf < MIN_OCR_CONFIDENCE:
        warnings.append(f"confidence_below_gate:{conf:.3f}")

    if not text.strip():
        warnings.append("empty_ocr_output")
        return warnings

    alnum = _alphanumeric_ratio(text)
    vc = _valid_char_ratio(text)
    ta_r = _tamil_ratio(text)
    la_r = _latin_ratio(text)

    if alnum < 0.30:
        warnings.append(f"low_alphanumeric_ratio:{alnum:.3f}")
    if vc < 0.40:
        warnings.append(f"low_valid_char_ratio:{vc:.3f}")

    # Script-mode mismatch
    if mode == ["ta"] and ta_r < 0.05:
        warnings.append("tamil_ocr_no_tamil_chars")
    if mode == ["en"] and ta_r > 0.50:
        warnings.append("english_ocr_high_tamil_ratio")

    # Very short text with multiple detections is suspicious
    det = int(result.get("detection_count", 0))
    if det > 5 and len(text.strip()) < 10:
        warnings.append("suspiciously_short_text_for_detection_count")

    return warnings


def evaluate_candidate(result: dict) -> dict:
    """
    Enrich an OCR result dict with quality metrics.

    Adds / updates:
        - usable
        - selection_score
        - warnings
        - tamil_ratio
        - latin_ratio
        - alphanumeric_ratio
        - valid_char_ratio
        - script_distribution

    Does NOT modify the candidate text or confidence values.
    """
    text = result.get("text", "")
    dist = script_distribution(text)
    result["script_distribution"] = dist
    result["tamil_ratio"] = dist.get("Tamil", 0.0)
    result["latin_ratio"] = dist.get("Latin", 0.0)
    result["alphanumeric_ratio"] = _alphanumeric_ratio(text)
    result["valid_char_ratio"] = _valid_char_ratio(text)
    result["usable"] = is_ocr_usable(result)
    result["selection_score"] = selection_score(result)
    result["warnings"] = quality_warnings(result)
    return result
