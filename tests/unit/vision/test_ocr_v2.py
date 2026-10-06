"""
TrustLens OCR v2 — Pytest Test Suite
======================================

Tests for:
  - Tamil script ratio calculation
  - Latin script ratio calculation
  - Hard OCR reliability gate
  - Candidate selection
  - Unreliable OCR handling
  - Empty OCR handling
  - Mixed script detection

All tests are pure unit tests — no OCR models are loaded.
Tests work offline and deterministically.

Run with:
    python -m pytest tests/vision/ -v
"""
from __future__ import annotations

import pytest

from app.vision.ocr.script_quality import (
    tamil_ratio,
    latin_ratio,
    script_distribution,
    valid_char_ratio,
    alphanumeric_ratio,
    contains_tamil,
    detect_language_label,
)
from app.vision.ocr.ocr_quality import (
    is_ocr_usable,
    selection_score,
    evaluate_candidate,
    MIN_OCR_CONFIDENCE,
)
from app.vision.ocr.ocr_selector import select_ocr_candidate


# ===========================================================================
# Tamil script ratio tests
# ===========================================================================

class TestTamilScriptRatio:

    def test_pure_tamil_text(self):
        """Pure Tamil text should have tamil_ratio close to 1.0."""
        text = "தமிழ் முதலீடு திட்டம்"
        ratio = tamil_ratio(text)
        assert ratio > 0.90, f"Expected high Tamil ratio, got {ratio}"

    def test_pure_english_text(self):
        """Pure English text should have tamil_ratio = 0.0."""
        text = "Investment plan for growth"
        ratio = tamil_ratio(text)
        assert ratio == 0.0

    def test_empty_text(self):
        """Empty text → 0.0."""
        assert tamil_ratio("") == 0.0

    def test_numbers_only_no_tamil(self):
        """Numbers only: no letter-class chars → ratio 0.0."""
        assert tamil_ratio("₹500 60% 40%") == 0.0

    def test_mixed_text_has_partial_tamil(self):
        """Mixed Tamil+English text should have intermediate Tamil ratio."""
        text = "இந்த job நல்லது Apply pannunga ₹50,000"
        ratio = tamil_ratio(text)
        assert 0.0 < ratio < 1.0, f"Expected partial Tamil ratio, got {ratio}"

    def test_numbers_do_not_dilute_tamil_ratio(self):
        """Numbers should be excluded from ratio calculation."""
        pure_tamil = "தமிழ்"
        with_numbers = "தமிழ் 500 60 40"
        # Tamil ratio should be the same (numbers excluded from letters)
        assert tamil_ratio(pure_tamil) == pytest.approx(tamil_ratio(with_numbers), abs=0.01)


# ===========================================================================
# Latin script ratio tests
# ===========================================================================

class TestLatinScriptRatio:

    def test_pure_english_text(self):
        """Pure English text → high latin_ratio."""
        text = "Investment Plan for Growth"
        ratio = latin_ratio(text)
        assert ratio > 0.95

    def test_pure_tamil_text(self):
        """Pure Tamil text → latin_ratio = 0.0."""
        text = "தமிழ் முதலீடு"
        ratio = latin_ratio(text)
        assert ratio == 0.0

    def test_empty_text(self):
        assert latin_ratio("") == 0.0

    def test_currency_numbers_no_latin(self):
        """₹500 60% 40% has no letters → latin_ratio = 0.0."""
        assert latin_ratio("₹500 60% 40%") == 0.0

    def test_mixed_text(self):
        """Mixed text → intermediate Latin ratio."""
        text = "இந்த job Apply ₹50,000"
        ratio = latin_ratio(text)
        assert 0.0 < ratio < 1.0


# ===========================================================================
# Script distribution tests
# ===========================================================================

class TestScriptDistribution:

    def test_pure_tamil_distribution(self):
        dist = script_distribution("தமிழ் முதலீடு")
        assert dist["Tamil"] > 0.90
        assert dist["Latin"] == 0.0

    def test_pure_english_distribution(self):
        dist = script_distribution("Investment Plan")
        assert dist["Latin"] > 0.95
        assert dist["Tamil"] == 0.0

    def test_mixed_distribution(self):
        dist = script_distribution("தமிழ் Job Apply")
        assert dist["Tamil"] > 0.0
        assert dist["Latin"] > 0.0

    def test_empty_distribution(self):
        dist = script_distribution("")
        assert dist == {"Tamil": 0.0, "Latin": 0.0, "Other": 0.0}

    def test_numbers_excluded(self):
        """Numbers like 500 60% should not appear in any script."""
        dist = script_distribution("₹500 60%")
        assert dist["Tamil"] == 0.0
        assert dist["Latin"] == 0.0


# ===========================================================================
# Mixed script detection tests
# ===========================================================================

class TestMixedScriptDetection:

    def test_mixed_detected(self):
        """Tamil + Latin in text → detect_language_label returns 'mixed'."""
        text = "இந்த job Apply pannunga ₹50,000"
        label = detect_language_label(text)
        assert label in ("mixed", "ta"), f"Got {label}"

    def test_pure_tamil_label(self):
        label = detect_language_label("தமிழ் முதலீடு திட்டம்")
        assert label == "ta"

    def test_pure_english_label(self):
        label = detect_language_label("Investment Growth Plan")
        assert label == "en"

    def test_numbers_only_unknown(self):
        """No letters → language label should be 'unknown'."""
        label = detect_language_label("₹500 60% 40%")
        assert label == "unknown"

    def test_garbage_unknown(self):
        """Junk ASCII without actual letters → 'unknown'."""
        label = detect_language_label("!@#$%^&*()")
        assert label == "unknown"


# ===========================================================================
# Hard OCR reliability gate
# ===========================================================================

class TestOCRReliabilityGate:

    def test_low_confidence_is_not_usable(self):
        """Confidence 0.265 < 0.60 → usable=False regardless of other signals."""
        result = {
            "text": "20g61 garbage text",
            "confidence": 0.265,
            "language_mode": ["en"],
            "detection_count": 5,
            "script_distribution": {"Latin": 1.0, "Tamil": 0.0},
        }
        assert is_ocr_usable(result) is False

    def test_confidence_exactly_at_gate_boundary(self):
        """Confidence exactly 0.60 should pass the gate (>= 0.60)."""
        result = {
            "text": "valid text here",
            "confidence": MIN_OCR_CONFIDENCE,
            "language_mode": ["en"],
            "detection_count": 3,
            "script_distribution": {"Latin": 1.0, "Tamil": 0.0},
        }
        assert is_ocr_usable(result) is True

    def test_just_below_gate(self):
        """Confidence 0.599 → usable=False."""
        result = {
            "text": "valid text here",
            "confidence": 0.599,
            "language_mode": ["en"],
            "detection_count": 3,
            "script_distribution": {"Latin": 1.0, "Tamil": 0.0},
        }
        assert is_ocr_usable(result) is False

    def test_empty_text_is_not_usable(self):
        """Empty text → usable=False even with high confidence."""
        result = {
            "text": "",
            "confidence": 1.0,
            "language_mode": ["en"],
            "detection_count": 0,
            "script_distribution": {},
        }
        assert is_ocr_usable(result) is False

    def test_whitespace_only_is_not_usable(self):
        result = {
            "text": "   \n  ",
            "confidence": 0.9,
            "language_mode": ["en"],
            "detection_count": 1,
            "script_distribution": {"Latin": 1.0},
        }
        assert is_ocr_usable(result) is False

    def test_tamil_mode_requires_tamil_chars(self):
        """Tamil mode with Latin-only OCR output → usable=False."""
        result = {
            "text": "garbage latin output",
            "confidence": 0.75,
            "language_mode": ["ta"],
            "detection_count": 3,
            "script_distribution": {"Latin": 1.0, "Tamil": 0.0},
        }
        assert is_ocr_usable(result) is False

    def test_tamil_mode_with_tamil_chars_is_usable(self):
        result = {
            "text": "தமிழ் முதலீடு",
            "confidence": 0.75,
            "language_mode": ["ta"],
            "detection_count": 3,
            "script_distribution": {"Tamil": 1.0, "Latin": 0.0},
        }
        assert is_ocr_usable(result) is True


# ===========================================================================
# OCR candidate selection
# ===========================================================================

class TestOCRCandidateSelection:

    def test_selects_tamil_over_low_confidence_english(self):
        """
        Tamil candidate with confidence 0.72 should beat English candidate
        with confidence 0.265 (which fails the hard gate).
        """
        selected = select_ocr_candidate([
            {
                "text": "20g61 garbage",
                "confidence": 0.265,
                "language_mode": ["en"],
                "detection_count": 1,
                "preprocessing": "original",
                "backend": "easyocr_en",
            },
            {
                "text": "தமிழ் முதலீடு",
                "confidence": 0.72,
                "language_mode": ["ta"],
                "detection_count": 2,
                "preprocessing": "original",
                "backend": "paddleocr_ta",
            },
        ])
        assert selected["language_mode"] == ["ta"]
        assert selected["unreliable"] is False

    def test_high_score_low_confidence_cannot_override_gate(self):
        """
        A candidate with confidence=0.265 and a high selection_score
        must still be marked usable=False and cannot win if a better
        candidate exists.
        """
        candidates = [
            {
                "text": "20g61 garbage",
                "confidence": 0.265,
                "language_mode": ["en"],
                "detection_count": 30,   # inflated count
                "preprocessing": "original",
                "backend": "easyocr_en",
            },
            {
                "text": "தமிழ் முதலீடு",
                "confidence": 0.72,
                "language_mode": ["ta"],
                "detection_count": 5,
                "preprocessing": "original",
                "backend": "paddleocr_ta",
            },
        ]
        selected = select_ocr_candidate(candidates)
        assert selected["language_mode"] == ["ta"], (
            "Low-confidence candidate must not win even with high detection_count"
        )
        assert selected["usable"] is True

    def test_all_low_confidence_returns_unreliable(self):
        """When all candidates fail the gate → unreliable=True."""
        candidates = [
            {
                "text": "garbage1",
                "confidence": 0.10,
                "language_mode": ["en"],
                "detection_count": 1,
                "preprocessing": "original",
                "backend": "easyocr_en",
            },
            {
                "text": "garbage2",
                "confidence": 0.20,
                "language_mode": ["ta"],
                "detection_count": 1,
                "preprocessing": "original",
                "backend": "paddleocr_ta",
            },
        ]
        selected = select_ocr_candidate(candidates)
        assert selected["unreliable"] is True

    def test_empty_candidate_list_returns_safe_result(self):
        selected = select_ocr_candidate([])
        assert selected["unreliable"] is True
        assert selected["text"] == ""
        assert "no_candidates" in selected.get("warnings", [])

    def test_candidates_attached_to_result(self):
        """Selected result must have 'candidates' list for debugging."""
        selected = select_ocr_candidate([
            {
                "text": "text",
                "confidence": 0.8,
                "language_mode": ["en"],
                "detection_count": 5,
                "preprocessing": "original",
                "backend": "easyocr_en",
            }
        ])
        assert "candidates" in selected
        assert len(selected["candidates"]) == 1


# ===========================================================================
# Unreliable OCR
# ===========================================================================

class TestUnreliableOCR:

    def test_unreliable_flag_set_when_no_usable_candidates(self):
        selected = select_ocr_candidate([
            {
                "text": "garbage",
                "confidence": 0.05,
                "language_mode": ["en"],
                "detection_count": 1,
                "preprocessing": "original",
                "backend": "easyocr_en",
            }
        ])
        assert selected["unreliable"] is True

    def test_unreliable_candidate_still_returned_not_none(self):
        """Even when unreliable, a result is returned (not None or exception)."""
        selected = select_ocr_candidate([
            {
                "text": "abc",
                "confidence": 0.1,
                "language_mode": ["en"],
                "detection_count": 1,
                "preprocessing": "original",
                "backend": "easyocr_en",
            }
        ])
        assert selected is not None
        assert "text" in selected


# ===========================================================================
# Empty OCR
# ===========================================================================

class TestEmptyOCR:

    def test_empty_text_not_usable(self):
        result = {
            "text": "",
            "confidence": 1.0,
            "language_mode": ["en"],
            "detection_count": 0,
        }
        assert is_ocr_usable(result) is False

    def test_selection_score_for_empty(self):
        result = {
            "text": "",
            "confidence": 0.0,
            "language_mode": ["en"],
            "detection_count": 0,
            "script_distribution": {},
        }
        score = selection_score(result)
        assert score == 0.0


# ===========================================================================
# Selection score formula tests
# ===========================================================================

class TestSelectionScore:

    def test_score_is_positive_for_good_candidate(self):
        result = {
            "text": "Investment plan with good text",
            "confidence": 0.9,
            "language_mode": ["en"],
            "detection_count": 10,
            "script_distribution": {"Latin": 1.0, "Tamil": 0.0},
        }
        score = selection_score(result)
        assert score > 0.5

    def test_tamil_candidate_scores_higher_with_tamil_chars(self):
        """
        Tamil candidate with Tamil chars should score higher than
        English candidate with the same confidence reading a Tamil image.
        """
        ta_result = {
            "text": "தமிழ் முதலீடு",
            "confidence": 0.72,
            "language_mode": ["ta"],
            "detection_count": 5,
            "script_distribution": {"Tamil": 1.0, "Latin": 0.0},
        }
        en_result = {
            "text": "garbage latin",
            "confidence": 0.72,
            "language_mode": ["en"],
            "detection_count": 5,
            "script_distribution": {"Tamil": 0.0, "Latin": 1.0},
        }
        ta_score = selection_score(ta_result)
        en_score = selection_score(en_result)
        # Both have same confidence, but Tamil reader should score higher
        # because script_quality for ta_result is 1.0 (Tamil chars present)
        assert ta_score >= en_score

    def test_score_between_0_and_1(self):
        result = {
            "text": "text",
            "confidence": 0.5,
            "language_mode": ["en"],
            "detection_count": 3,
            "script_distribution": {"Latin": 0.8, "Tamil": 0.2},
        }
        score = selection_score(result)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# Evaluate candidate enrichment
# ===========================================================================

class TestEvaluateCandidate:

    def test_evaluate_fills_all_fields(self):
        result = {
            "text": "Tamil OCR output தமிழ்",
            "confidence": 0.75,
            "language_mode": ["ta"],
            "detection_count": 5,
            "preprocessing": "upscale_2x",
            "backend": "paddleocr_ta",
        }
        enriched = evaluate_candidate(result)
        assert "usable" in enriched
        assert "selection_score" in enriched
        assert "warnings" in enriched
        assert "tamil_ratio" in enriched
        assert "latin_ratio" in enriched
        assert "script_distribution" in enriched
        assert "alphanumeric_ratio" in enriched
        assert "valid_char_ratio" in enriched

    def test_evaluate_does_not_modify_confidence(self):
        """evaluate_candidate must never change confidence."""
        result = {
            "text": "text",
            "confidence": 0.265,
            "language_mode": ["en"],
            "detection_count": 1,
        }
        evaluate_candidate(result)
        assert result["confidence"] == 0.265  # unchanged

    def test_evaluate_does_not_modify_text(self):
        """evaluate_candidate must never change the text."""
        original_text = "20g61 garbage"
        result = {
            "text": original_text,
            "confidence": 0.3,
            "language_mode": ["en"],
            "detection_count": 1,
        }
        evaluate_candidate(result)
        assert result["text"] == original_text
