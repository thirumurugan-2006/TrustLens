"""
TrustLens Step 11 — Rule-Based Reliability Scorer
===================================================

Produces a calibrated reliability score from measurable signals WITHOUT
using ground-truth text as input.

The weights below were selected after analysing the Tamil benchmark dataset.
Weight justification is documented in reports/tamil_confidence_analysis.md.

Signal weights (must sum to 1.0):
  - raw_confidence  : 0.40  (primary engine signal, but poorly calibrated for Tamil)
  - tamil_ratio     : 0.20  (script authenticity — detects garbled latin output)
  - valid_char_ratio: 0.20  (detects junk characters from OCR noise)
  - length_score    : 0.10  (very short outputs are less reliable)
  - region_score    : 0.10  (more consistent regions → higher reliability)

These weights were NOT chosen arbitrarily. The analysis in Step 11 showed:
  - raw_confidence alone has weak monotonic relationship with CER for Tamil
  - tamil_ratio is a strong binary signal (if output is not Tamil, OCR failed)
  - valid_char_ratio catches partial garbling
  - length and region scores break ties and handle degenerate cases
"""
from __future__ import annotations

from app.vision.ocr.tamil_script import tamil_ratio, valid_char_ratio

# Weights — documented and validated against Step 11 benchmark
_W_CONFIDENCE = 0.40
_W_TAMIL_RATIO = 0.20
_W_VALID_CHAR = 0.20
_W_LENGTH = 0.10
_W_REGION = 0.10

# Operational threshold: below this, result is marked unreliable
RELIABILITY_THRESHOLD = 0.45

# Normalisation constants
_MAX_EXPECTED_LENGTH = 200   # characters beyond this score full length points
_MAX_EXPECTED_REGIONS = 15   # regions beyond this score full region points


def compute_reliability_score(
    text: str,
    raw_confidence: float | None,
    region_confidences: list[float],
    backend: str = "unknown",
    expected_language: str = "ta",
) -> float:
    """
    Compute a TrustLens reliability score from measurable signals.

    Parameters
    ----------
    text : str
        OCR output text.
    raw_confidence : float | None
        Average raw confidence from the OCR engine (0.0–1.0).
    region_confidences : list[float]
        Per-region confidence values from the OCR engine.
    backend : str
        Name of the OCR backend (informational only).
    expected_language : str
        Expected language code ("ta" or "en").

    Returns
    -------
    float
        Reliability score in [0.0, 1.0].

    Notes
    -----
    Ground-truth text is NOT used here — this function is safe to call
    during production inference.
    """
    # ── Signal 1: raw confidence ─────────────────────────────────────────────
    conf_score = float(raw_confidence) if raw_confidence is not None else 0.0
    conf_score = max(0.0, min(1.0, conf_score))

    # ── Signal 2: Tamil character ratio ─────────────────────────────────────
    if expected_language == "ta":
        ta_score = tamil_ratio(text)
    else:
        # For English-mode, penalise unexpected Tamil characters
        ta_r = tamil_ratio(text)
        ta_score = 1.0 - ta_r  # full score when no Tamil present

    # ── Signal 3: valid character ratio ─────────────────────────────────────
    vc_score = valid_char_ratio(text)

    # ── Signal 4: text length score ──────────────────────────────────────────
    length = len(text.strip())
    if length == 0:
        len_score = 0.0
    else:
        len_score = min(length / _MAX_EXPECTED_LENGTH, 1.0)

    # ── Signal 5: region consistency score ───────────────────────────────────
    n_regions = len(region_confidences)
    if n_regions == 0:
        reg_score = 0.0
    else:
        # Combine region count normalisation with confidence consistency
        count_norm = min(n_regions / _MAX_EXPECTED_REGIONS, 1.0)
        avg_reg_conf = sum(region_confidences) / n_regions
        # Reward high average AND having multiple regions
        reg_score = 0.6 * avg_reg_conf + 0.4 * count_norm

    reliability = (
        _W_CONFIDENCE * conf_score
        + _W_TAMIL_RATIO * ta_score
        + _W_VALID_CHAR * vc_score
        + _W_LENGTH * len_score
        + _W_REGION * reg_score
    )
    return round(min(max(reliability, 0.0), 1.0), 4)


def is_reliable(reliability_score: float) -> bool:
    """Return True if the reliability score meets the operational threshold."""
    return reliability_score >= RELIABILITY_THRESHOLD


def ocr_status_from_result(
    text: str,
    raw_confidence: float | None,
    reliability_score: float,
    error: str = "",
) -> str:
    """
    Map OCR result signals to a canonical TrustLens OCR status string.

    Statuses:
      SUCCESS          — text present and reliability meets threshold
      PARTIAL          — text present but short (< 3 characters after strip)
      LOW_CONFIDENCE   — text present but reliability below threshold
      LANGUAGE_UNCERTAIN — no clear script evidence
      OCR_FAILED       — no text or an error occurred
    """
    from app.vision.ocr.tamil_script import tamil_ratio as ta_ratio, valid_char_ratio as vc_ratio

    if error or not text or not text.strip():
        return "OCR_FAILED"

    ta_r = ta_ratio(text)
    vc_r = vc_ratio(text)
    stripped = text.strip()

    # Very short output is suspicious
    if len(stripped) < 3:
        return "PARTIAL"

    # No recognisable script at all
    if ta_r == 0.0 and vc_r < 0.10:
        return "LANGUAGE_UNCERTAIN"

    if not is_reliable(reliability_score):
        return "LOW_CONFIDENCE"

    return "SUCCESS"
