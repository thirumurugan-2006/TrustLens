"""
TrustLens OCR v2 — Full Pipeline
==================================

OCRPipeline orchestrates the complete OCR v2 workflow:

    Screenshot
        ↓
    Image Validation
        ↓
    OCRPreprocessor  →  6 variants
        ↓
    Language/Script Hints  (cheap scan to decide which readers to run)
        ↓
    For each relevant (variant × language) pair:
        English OCR  (EasyOCR["en"])
        Tamil OCR    (PaddleOCR["ta"])
        ↓
    OCR Quality Evaluation  (evaluate_candidate)
        ↓
    Candidate Ranking  (select_ocr_candidate)
        ↓
    Best OCRResult
        ↓
    Unicode normalization
        ↓
    Language Analysis

Architecture rules
------------------
- easyocr.Reader(["en", "ta"]) is NEVER used.
- Readers are lazily loaded once via OCRService.
- Preprocessing variants are written to temp files, read by both engines,
  then deleted.
- The hard confidence gate (0.60) is enforced by the quality evaluator.
- A high selection_score on a low-confidence result cannot win.
- Raw OCR text is preserved separately from normalized text.
- OCR failure never raises an unhandled exception → unreliable=True response.

Candidate generation strategy
------------------------------
Phase 1  (always):
    Run English OCR on "original" image.
    If result is usable → still continue unless fast_mode=True.

Phase 2  (always for correctness, unless fast_mode=True):
    Run English + Tamil OCR on:
        - original
        - upscale_2x
        - grayscale_upscale_2x
        - contrast_enhanced
        - adaptive_threshold
    (grayscale = upscale_2x grayscale-version, so reused)

The first implementation prioritizes correctness over speed.
A fast_mode option can be added later to skip variants when a quick
cheap hint strongly indicates one language.
"""
from __future__ import annotations

import logging
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Optional

from app.vision.ocr.ocr_engine import OCRService
from app.vision.ocr.ocr_preprocessor import OCRPreprocessor
from app.vision.ocr.ocr_quality import evaluate_candidate
from app.vision.ocr.ocr_selector import select_ocr_candidate
from app.vision.ocr.schemas import OCRCandidate, OCRResult
from app.vision.ocr.script_quality import (
    detect_language_label,
    script_distribution,
    tamil_ratio,
    latin_ratio,
    valid_char_ratio,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version tag
# ---------------------------------------------------------------------------
OCR_V2_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Preprocessing variants used in the full pipeline
# ---------------------------------------------------------------------------
# Each tuple is (variant_name, run_english, run_tamil)
# "english_hint_only" means: run English first on "original" cheaply
_FULL_VARIANT_PLAN = [
    ("original",             True,  True),
    ("upscale_2x",           True,  True),
    ("grayscale_upscale_2x", True,  True),
    ("contrast_enhanced",    False, True),   # mainly useful for Tamil
    ("adaptive_threshold",   False, True),   # mainly useful for Tamil
]

# For fast English-only mode (used when cheap scan shows no Tamil chars)
_ENGLISH_VARIANT_PLAN = [
    ("upscale_2x",           True,  False),
    ("grayscale_upscale_2x", True,  False),
    ("contrast_enhanced",    True,  False),
]


class OCRPipeline:
    """
    Full OCR v2 pipeline.

    Parameters
    ----------
    model_directory : str | Path
        Directory where EasyOCR model weights are cached.
        Default: "data/models/easyocr"
    fast_mode : bool
        If True, skip Tamil OCR when a cheap English scan looks usable.
        Default: False (correctness mode).
    """

    def __init__(
        self,
        model_directory: str | Path = "data/models/easyocr",
        fast_mode: bool = False,
    ) -> None:
        self.service = OCRService(model_directory)
        self.preprocessor = OCRPreprocessor()
        self.fast_mode = fast_mode

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(self, image_path: str | Path) -> OCRResult:
        """
        Run the full OCR v2 pipeline on an image.

        Returns
        -------
        OCRResult
            Structured result with selected candidate, all candidates for
            debugging, and normalized / raw text.
        """
        image_path = Path(image_path)
        candidates_raw: list[dict] = []
        tmp_files: list[Path] = []
        attempts = 0

        try:
            # ── Step 1: generate preprocessing variants ──────────────────
            try:
                variant_files = self.preprocessor.variants_as_files(image_path)
                tmp_files.extend(p for _, p in variant_files)
            except Exception as exc:
                logger.error("Preprocessing failed for %s: %s", image_path, exc)
                return self._error_result(str(exc), image_path)

            variant_map: dict[str, Path] = {name: path for name, path in variant_files}

            # ── Step 2: cheap script hint (English-only scan) ────────────
            original_path = variant_map.get("original", image_path)
            quick_en = self._quick_english_scan(original_path)
            candidates_raw.append(quick_en)
            attempts += 1
            
            has_tamil_hint = quick_en.get("tamil_ratio", 0.0) > 0.05
            english_usable = quick_en.get("usable", False)
            english_conf = quick_en.get("confidence", 0.0)

            variant_plan = []
            en_available = self.service.english_available()
            ta_available = self.service.tamil_available()

            # Fast path 1: Excellent English OCR
            if english_usable and not has_tamil_hint and english_conf >= 0.80:
                pass  # Skip variant plan
            else:
                # Fast path 2: Excellent Tamil OCR on original
                ta_done = False
                if (has_tamil_hint or not english_usable) and ta_available:
                    quick_ta = self._quick_tamil_scan(original_path)
                    candidates_raw.append(quick_ta)
                    attempts += 1
                    ta_done = True
                    ta_usable = quick_ta.get("usable", False)
                    ta_conf = quick_ta.get("confidence", 0.0)
                    
                    if ta_usable and ta_conf >= 0.80:
                        variant_plan = []  # Skip rest

                # If fast paths didn't skip, use the appropriate plan
                if not variant_plan and not (ta_done and quick_ta.get("confidence", 0.0) >= 0.80):
                    if self.fast_mode and english_usable and not has_tamil_hint:
                        variant_plan = _ENGLISH_VARIANT_PLAN
                    else:
                        variant_plan = _FULL_VARIANT_PLAN

            # ── Step 3: run OCR candidates ───────────────────────────────
            for variant_name, run_en, run_ta in variant_plan:
                variant_path = variant_map.get(variant_name)
                if variant_path is None:
                    continue

                if run_en and en_available:
                    if variant_name == "original":
                        continue  # Already ran
                    try:
                        raw = self.service.run_ocr(variant_path, ["en"], variant_name)
                        evaluate_candidate(raw)
                        candidates_raw.append(raw)
                        attempts += 1
                    except Exception as exc:
                        logger.warning("English OCR on %s failed: %s", variant_name, exc)

                if run_ta and ta_available:
                    if variant_name == "original" and 'ta_done' in locals() and ta_done:
                        continue  # Already ran
                    try:
                        raw = self.service.run_ocr(variant_path, ["ta"], variant_name)
                        evaluate_candidate(raw)
                        candidates_raw.append(raw)
                        attempts += 1
                    except Exception as exc:
                        logger.warning("Tamil OCR on %s failed: %s", variant_name, exc)

            # ── Step 4: select best candidate ────────────────────────────
            if not candidates_raw:
                result = self._no_candidate_result(en_available, ta_available, image_path)
                result.attempts = attempts
                return result

            selected = select_ocr_candidate(candidates_raw)
            selected["attempts"] = attempts

            # ── Step 5: normalize text ───────────────────────────────────
            raw_text = selected.get("text", "")
            norm_text = _normalize_text(raw_text)

            # ── Step 6: build OCRResult ──────────────────────────────────
            dist = selected.get("script_distribution") or script_distribution(raw_text)
            ta_r = dist.get("Tamil", 0.0)
            la_r = dist.get("Latin", 0.0)

            warnings: list[str] = list(selected.get("warnings", []))
            if not self.service.tamil_available():
                warnings.append("tamil_ocr_unavailable")
            if not self.service.english_available():
                warnings.append("english_ocr_unavailable")

            return OCRResult(
                text=norm_text,
                raw_text=raw_text,
                normalized_text=norm_text,
                confidence=selected.get("confidence", 0.0),
                language_mode=selected.get("language_mode", []),
                preprocessing=selected.get("preprocessing", "original"),
                backend=selected.get("backend", "unknown"),
                detection_count=selected.get("detection_count", 0),
                text_length=len(norm_text),
                script_distribution=dist,
                tamil_ratio=ta_r,
                latin_ratio=la_r,
                selection_score=selected.get("selection_score", 0.0),
                usable=selected.get("usable", False),
                unreliable=selected.get("unreliable", True),
                warnings=warnings,
                candidates=self._build_candidate_models(selected.get("candidates", [])),
                attempts=attempts,
                ocr_engine="easyocr_en+paddleocr_ta",
                ocr_version=self._get_easyocr_version(),
            )

        finally:
            # ── Cleanup temp files ───────────────────────────────────────
            for tmp in tmp_files:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quick_english_scan(self, image_path: Path) -> dict:
        """
        Run a cheap English OCR scan on the original image to get a
        script hint without loading Tamil models.
        """
        if not self.service.english_available():
            return {"tamil_ratio": 0.0, "usable": False, "confidence": 0.0, "backend": "unknown", "warnings": ["english_ocr_unavailable"]}
        try:
            raw = self.service.run_ocr(image_path, ["en"], "original")
            evaluate_candidate(raw)
            return raw
        except Exception as exc:
            return {"tamil_ratio": 0.0, "usable": False, "confidence": 0.0, "warnings": [f"error:{exc}"]}

    def _quick_tamil_scan(self, image_path: Path) -> dict:
        """
        Run a cheap Tamil OCR scan on the original image.
        """
        if not self.service.tamil_available():
            return {"usable": False, "confidence": 0.0, "backend": "unknown", "warnings": ["tamil_ocr_unavailable"]}
        try:
            raw = self.service.run_ocr(image_path, ["ta"], "original")
            evaluate_candidate(raw)
            return raw
        except Exception as exc:
            return {"usable": False, "confidence": 0.0, "warnings": [f"error:{exc}"]}

    def _build_candidate_models(self, raw_candidates: list[dict]) -> list[OCRCandidate]:
        """Convert raw candidate dicts to OCRCandidate Pydantic models."""
        models: list[OCRCandidate] = []
        for c in raw_candidates:
            models.append(OCRCandidate(
                language_mode=c.get("language_mode", []),
                preprocessing=c.get("preprocessing", "original"),
                confidence=c.get("confidence", 0.0),
                detection_count=c.get("detection_count", 0),
                text_length=c.get("text_length", len(c.get("text", ""))),
                tamil_ratio=c.get("tamil_ratio", 0.0),
                latin_ratio=c.get("latin_ratio", 0.0),
                script_distribution=c.get("script_distribution", {}),
                alphanumeric_ratio=c.get("alphanumeric_ratio", 0.0),
                valid_char_ratio=c.get("valid_char_ratio", 0.0),
                selection_score=c.get("selection_score", 0.0),
                usable=c.get("usable", False),
                warnings=c.get("warnings", []),
                text=c.get("text", ""),
                backend=c.get("backend", "unknown"),
            ))
        return models

    def _error_result(self, error_msg: str, image_path: Path) -> OCRResult:
        """Return a safe error OCRResult without raising."""
        return OCRResult(
            unreliable=True,
            warnings=[f"pipeline_error:{error_msg}"],
            ocr_engine="easyocr_en+paddleocr_ta",
            ocr_version=self._get_easyocr_version(),
        )

    def _no_candidate_result(
        self,
        en_available: bool,
        ta_available: bool,
        image_path: Path,
    ) -> OCRResult:
        """Return a safe empty OCRResult when no candidates were generated."""
        warnings: list[str] = ["no_ocr_candidates_generated"]
        if not en_available:
            warnings.append("english_ocr_unavailable")
        if not ta_available:
            warnings.append("tamil_ocr_unavailable")
        return OCRResult(
            unreliable=True,
            warnings=warnings,
            ocr_engine="easyocr_en+paddleocr_ta",
            ocr_version=self._get_easyocr_version(),
        )

    @staticmethod
    def _get_easyocr_version() -> str:
        try:
            import easyocr
            return getattr(easyocr, "__version__", "unknown")
        except ImportError:
            return "not_installed"


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """
    Apply minimal, safe Unicode normalization.

    Steps:
      1. NFC normalization (compose precomposed Unicode sequences)
      2. Remove ASCII control characters (keep newline/tab)
      3. Collapse consecutive whitespace
      4. Strip leading/trailing whitespace
    """
    if not text:
        return ""
    # NFC compose
    text = unicodedata.normalize("NFC", text)
    # Remove control characters except space, tab, newline
    cleaned = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf") or ch in (" ", "\t", "\n")
    )
    # Collapse whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()
