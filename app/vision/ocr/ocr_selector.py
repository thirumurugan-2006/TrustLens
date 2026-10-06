"""
TrustLens OCR v2 — Candidate Selector
=======================================

Selects the best OCR candidate from a list of evaluated candidates.

Selection pipeline
------------------
1. Remove empty results (no text at all).
2. Apply hard reliability gate via is_ocr_usable().
3. Evaluate selection_score for all candidates.
4. Select the highest-scoring usable candidate.
5. If no usable candidate exists → fall back to the highest-scoring
   candidate but mark unreliable=True.

Rules
-----
- A candidate with confidence < 0.60 is NEVER selected as the winner
  unless it is the only candidate (fallback mode).
- The fallback (unreliable) result must still be returned so the API
  can emit useful diagnostic information.
- OCR failure must not produce HTTP 500.

Candidate metadata preserved for debugging
-------------------------------------------
All evaluated candidates are attached to the selected result under
the "candidates" key so engineers can inspect the full decision.
"""
from __future__ import annotations

from app.vision.ocr.ocr_quality import evaluate_candidate


def select_ocr_candidate(candidates: list[dict]) -> dict:
    """
    Select the best OCR candidate from a list of raw OCR result dicts.

    Each dict must contain at minimum:
        - text (str)
        - confidence (float)
        - language_mode (list[str])
        - detection_count (int)

    Returns
    -------
    dict
        The selected candidate enriched with:
            - usable (bool)
            - selection_score (float)
            - unreliable (bool)
            - candidates (list[dict])  — all evaluated candidates for debug
    """
    if not candidates:
        return _empty_result()

    # Step 1: Enrich all candidates with quality metrics
    for candidate in candidates:
        evaluate_candidate(candidate)

    # Step 2: Separate usable candidates (confidence gate passed)
    usable = [c for c in candidates if c["usable"]]

    # Step 3: Select
    if usable:
        selected = max(usable, key=lambda c: c["selection_score"])
        selected["unreliable"] = False
    else:
        # Fallback: return best non-empty candidate even if below gate
        non_empty = [c for c in candidates if c.get("text", "").strip()]
        pool = non_empty if non_empty else candidates
        selected = max(pool, key=lambda c: c["selection_score"])
        selected["unreliable"] = True

    # Step 4: Attach debug candidate table
    selected["candidates"] = [
        {
            "language_mode": c["language_mode"],
            "preprocessing": c.get("preprocessing", "original"),
            "confidence": c["confidence"],
            "detection_count": c.get("detection_count", 0),
            "text_length": len(c.get("text", "")),
            "tamil_ratio": c.get("tamil_ratio", 0.0),
            "latin_ratio": c.get("latin_ratio", 0.0),
            "selection_score": c["selection_score"],
            "usable": c["usable"],
            "warnings": c.get("warnings", []),
            "backend": c.get("backend", "unknown"),
        }
        for c in candidates
    ]

    return selected


def _empty_result() -> dict:
    """Return a safe empty result when no candidates are provided."""
    return {
        "text": "",
        "confidence": 0.0,
        "language_mode": [],
        "preprocessing": "original",
        "detection_count": 0,
        "text_length": 0,
        "tamil_ratio": 0.0,
        "latin_ratio": 0.0,
        "script_distribution": {},
        "alphanumeric_ratio": 0.0,
        "valid_char_ratio": 0.0,
        "selection_score": 0.0,
        "usable": False,
        "unreliable": True,
        "warnings": ["no_candidates"],
        "candidates": [],
        "backend": "unknown",
    }
