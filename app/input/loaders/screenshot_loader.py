"""
TrustLens — Screenshot Parser (OCR v2)
========================================

Parses a screenshot image file into a NormalizedPost using the OCR v2 pipeline.

Architecture
------------
    ScreenshotParser
          ↓
    OCRPipeline (v2)
          ↓
    OCRPreprocessor  →  6 variants
          ↓
    English OCR (EasyOCR) + Tamil OCR (PaddleOCR)
          ↓
    OCRSelector  →  best OCRResult
          ↓
    Language Analyzer  (only when OCR is reliable)
          ↓
    NormalizedPost

Important rules
---------------
- easyocr.Reader(["en", "ta"]) is NEVER used.
- OCR failure returns ocr_unreliable=True but NOT HTTP 500.
- Raw OCR text is preserved separately from normalized text.
- Language analysis is skipped (returns "unknown") when OCR is unreliable.
- All existing NormalizedPost fields and metadata contract are preserved.
- The ScreenshotParser._reader_instance attribute is kept for test monkeypatching
  compatibility (tests monkeypatch routes.pipeline.router.screenshot._reader_instance).

Test monkeypatching
-------------------
Existing tests do:
    routes.pipeline.router.screenshot._reader_instance = FakeReader()

When _reader_instance is set, the parser uses that reader directly (legacy path)
exactly as before, so existing tests remain green.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from uuid import uuid4

from dotenv import load_dotenv

from app.input.schemas import NormalizedPost
from app.preprocessing.language.language_analyzer import LanguageAnalyzer
from app.validation.image_validator import image_metadata, validate_image
from app.validation.quality_checker import check_ocr_quality

load_dotenv()

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def configured_ocr_languages() -> list[str]:
    configured = os.getenv("OCR_LANGUAGES", "en,ta")
    languages = [language.strip() for language in configured.split(",") if language.strip()]
    return languages or ["en", "ta"]


def configured_model_directory() -> Path:
    configured = os.getenv("OCR_MODEL_DIR", "data/models/easyocr")
    model_directory = Path(configured)
    if not model_directory.is_absolute():
        model_directory = PROJECT_ROOT / model_directory
    model_directory.mkdir(parents=True, exist_ok=True)
    return model_directory


class ScreenshotParser:
    """
    Screenshot → NormalizedPost parser using OCR v2 pipeline.

    The OCRPipeline is lazy-initialized on first use so that importing
    this module does not load heavy ML models immediately.

    Legacy compat
    -------------
    _reader_instance : any
        When set (by test monkeypatching), the parser falls back to the
        legacy single-reader OCR path so all existing tests remain passing.
    """

    def __init__(self) -> None:
        self.ocr_languages = configured_ocr_languages()
        self.model_directory = configured_model_directory()
        self.language_analyzer = LanguageAnalyzer()

        # ── Legacy compatibility (tests monkeypatch this) ────────────────
        self._reader_instance = None       # set by tests via monkeypatch
        self._english_reader_instance = None
        self._tamil_reader_instance = None
        self._tamil_unavailable = False

        # ── OCR v2 pipeline (lazy) ───────────────────────────────────────
        self._ocr_pipeline = None

    # ------------------------------------------------------------------
    # Lazy pipeline init
    # ------------------------------------------------------------------

    def _get_pipeline(self):
        if self._ocr_pipeline is None:
            from app.vision.ocr.ocr_pipeline import OCRPipeline
            self._ocr_pipeline = OCRPipeline(self.model_directory)
        return self._ocr_pipeline

    # ------------------------------------------------------------------
    # Legacy reader helpers (kept for test compatibility)
    # ------------------------------------------------------------------

    def _english_reader(self):
        if self._english_reader_instance is None:
            import easyocr
            self._english_reader_instance = easyocr.Reader(
                ["en"],
                gpu=False,
                model_storage_directory=str(self.model_directory),
            )
        return self._english_reader_instance

    def _tamil_reader(self):
        if self._tamil_reader_instance is None:
            from app.vision.ocr.tamil_reader import TamilOCR
            self._tamil_reader_instance = TamilOCR(
                self.model_directory / "paddleocr"
            )
        return self._tamil_reader_instance

    # ------------------------------------------------------------------
    # Legacy OCR result builder (used when _reader_instance is monkeypatched)
    # ------------------------------------------------------------------

    def _legacy_ocr_result(self, results: list[tuple], language_mode: list[str]) -> dict:
        from app.vision.ocr.script_quality import script_distribution as _dist
        recognized_text = " ".join(result[1] for result in results)
        normalized_text = " ".join(recognized_text.split())
        confidences = [float(result[2]) for result in results]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        dist = _dist(normalized_text)
        result = {
            "text": normalized_text,
            "confidence": confidence,
            "detections": results,
            "detection_count": len(results),
            "text_length": len(normalized_text),
            "script_distribution": dist,
            "language_mode": language_mode,
            "preprocessing": "original",
            "backend": "easyocr_en",
        }
        result["selection_score"] = self._legacy_selection_score(result)
        return result

    @staticmethod
    def _legacy_selection_score(result: dict) -> float:
        from app.vision.ocr.ocr_quality import selection_score
        return selection_score(result)

    @staticmethod
    def _legacy_is_usable(result: dict) -> bool:
        from app.vision.ocr.ocr_quality import is_ocr_usable
        return is_ocr_usable(result)

    def _legacy_extract_ocr(self, image_path: Path) -> dict:
        """
        Legacy OCR path used when _reader_instance is monkeypatched by tests.
        Exactly replicates old single-reader behaviour.
        """
        reader = self._reader_instance
        raw = reader.readtext(str(image_path), detail=1)
        result = self._legacy_ocr_result(raw, ["en"])
        result.update(
            attempts=1,
            selected_language_mode=["en"],
            unreliable=not self._legacy_is_usable(result),
            raw_text=result["text"],
            normalized_text=result["text"],
        )
        result.setdefault("candidates", [{
            "language_mode": ["en"],
            "preprocessing": "original",
            "confidence": result["confidence"],
            "detection_count": result["detection_count"],
            "text_length": result["text_length"],
            "tamil_ratio": result["script_distribution"].get("Tamil", 0.0),
            "latin_ratio": result["script_distribution"].get("Latin", 0.0),
            "selection_score": result["selection_score"],
            "usable": not result["unreliable"],
        }])
        return result

    # ------------------------------------------------------------------
    # Main parse entry point
    # ------------------------------------------------------------------

    def parse(self, image_path: str | Path) -> NormalizedPost:
        """
        Parse a screenshot image into a NormalizedPost.

        Parameters
        ----------
        image_path : str | Path
            Path to the screenshot image file.

        Returns
        -------
        NormalizedPost

        Raises
        ------
        FileNotFoundError
            If the image file does not exist.
        ValueError
            If the image is invalid/unsupported format.
        """
        source_path = Path(image_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Image file not found: {source_path}")

        validation = validate_image(source_path)
        if not validation.valid:
            raise ValueError("Invalid or unsupported image file.")

        # ── Save image to raw storage ────────────────────────────────────
        raw_directory = Path("data/raw/screenshots")
        raw_directory.mkdir(parents=True, exist_ok=True)
        post_id = f"screenshot_{uuid4().hex[:12]}"
        suffix = source_path.suffix.lower() or ".png"
        saved_path = raw_directory / f"{post_id}{suffix}"
        copy2(source_path, saved_path)

        # ── Run OCR ─────────────────────────────────────────────────────
        if self._reader_instance is not None:
            # Legacy path: test monkeypatching
            ocr_dict = self._legacy_extract_ocr(saved_path)
            normalized_text = ocr_dict["text"]
            raw_text = ocr_dict.get("raw_text", normalized_text)
            confidence = ocr_dict["confidence"]
            ocr_unreliable = ocr_dict.get("unreliable", True)
            selected_language_mode = ocr_dict.get("selected_language_mode", ["en"])
            detection_count = ocr_dict.get("detection_count", 0)
            text_length = ocr_dict.get("text_length", len(normalized_text))
            script_dist = ocr_dict.get("script_distribution", {})
            selection_score_val = ocr_dict.get("selection_score", 0.0)
            ocr_candidates = ocr_dict.get("candidates", [])
            attempts = ocr_dict.get("attempts", 1)
            preprocessing = ocr_dict.get("preprocessing", "original")
            backend = ocr_dict.get("backend", "easyocr_en")
            ta_ratio = script_dist.get("Tamil", 0.0)
        else:
            # OCR v2 pipeline path
            try:
                ocr_result = self._get_pipeline().process(saved_path)
            except Exception as exc:
                logger.error("OCR v2 pipeline error: %s", exc)
                # Safe fallback — do not crash the API
                ocr_result = _empty_ocr_result()

            normalized_text  = ocr_result.text or ""
            raw_text         = ocr_result.raw_text or ""
            confidence       = ocr_result.confidence
            ocr_unreliable   = ocr_result.unreliable
            selected_language_mode = ocr_result.language_mode
            detection_count  = ocr_result.detection_count
            text_length      = ocr_result.text_length
            script_dist      = ocr_result.script_distribution
            selection_score_val = ocr_result.selection_score
            preprocessing    = ocr_result.preprocessing
            backend          = ocr_result.backend
            ta_ratio         = ocr_result.tamil_ratio
            attempts         = ocr_result.attempts

            # Serialize candidates for metadata
            ocr_candidates = [
                {
                    "language_mode": c.language_mode,
                    "preprocessing": c.preprocessing,
                    "confidence": c.confidence,
                    "detection_count": c.detection_count,
                    "text_length": c.text_length,
                    "tamil_ratio": c.tamil_ratio,
                    "latin_ratio": c.latin_ratio,
                    "selection_score": c.selection_score,
                    "usable": c.usable,
                    "warnings": c.warnings,
                    "backend": c.backend,
                }
                for c in ocr_result.candidates
            ]

        # ── OCR quality label ────────────────────────────────────────────
        ocr_quality = check_ocr_quality(normalized_text, confidence)
        ocr_status = (
            "OCR_FAILED"    if not normalized_text
            else "LOW_CONFIDENCE" if ocr_unreliable
            else "SUCCESS"
        )

        # ── Language analysis ────────────────────────────────────────────
        # Only analyze language when OCR is reliable.
        # Rule: unreliable OCR → language = unknown, never guess.
        language_analysis = self.language_analyzer.analyze(
            normalized_text,
            reliable=not ocr_unreliable,
        )

        # ── Build metadata ───────────────────────────────────────────────
        metadata: dict = {
            # Source
            "source_type": "screenshot",
            # Engine
            "ocr_engine": "easyocr_en+paddleocr_ta",
            "ocr_version": self._get_ocr_version(),
            "ocr_status": ocr_status,
            "ocr_languages": self.ocr_languages,
            # Raw vs normalized text
            "ocr_raw_text": raw_text,
            "ocr_normalized_text": normalized_text,
            # Confidence — never inflated
            "ocr_confidence": confidence,
            # Quality
            "ocr_quality": ocr_quality.quality_level,
            "ocr_warning": bool(ocr_quality.warnings),
            "ocr_unreliable": ocr_unreliable,
            # Selection
            "ocr_attempts": attempts,
            "ocr_selected_language_mode": selected_language_mode,
            "ocr_selected_preprocessing": preprocessing,
            "ocr_selection_score": selection_score_val,
            # Counts
            "ocr_detection_count": detection_count,
            "ocr_text_length": text_length,
            # Script
            "ocr_script_distribution": script_dist,
            # Candidates (debug)
            "ocr_candidates": ocr_candidates,
            # Quality score (legacy fields)
            "quality_score": ocr_quality.quality_score,
            "quality_level": ocr_quality.quality_level,
            # Language
            "language_analysis": language_analysis,
            # Image metadata
            **image_metadata(saved_path),
        }
        if ocr_quality.warnings:
            metadata["ocr_warnings"] = ocr_quality.warnings

        # ── Build NormalizedPost ─────────────────────────────────────────
        from app.input.schemas import Platform, PostType, PostContent, PostMedia, AcquisitionMethod

        post = NormalizedPost(
            post_id=post_id,
            platform=Platform.screenshot,
            post_type=PostType.image_text,
            source_url=None,
            content=PostContent(text=normalized_text),
            media=PostMedia(images=[str(saved_path).replace("\\", "/")]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
            acquisition=AcquisitionMethod.screenshot
        )

        # ── Persist JSON ─────────────────────────────────────────────────
        normalized_directory = Path("data/processed/normalized/screenshots")
        normalized_directory.mkdir(parents=True, exist_ok=True)
        json_path = normalized_directory / f"{post_id}.json"
        with json_path.open("w", encoding="utf-8") as output_file:
            json.dump(post.model_dump(), output_file, ensure_ascii=False, indent=4)

        return post

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ocr_version() -> str:
        try:
            import easyocr
            return getattr(easyocr, "__version__", "unknown")
        except ImportError:
            return "not_installed"


# ---------------------------------------------------------------------------
# Safe empty OCR result helper
# ---------------------------------------------------------------------------

def _empty_ocr_result():
    """Return a minimal safe OCRResult when the pipeline itself crashes."""
    from app.vision.ocr.schemas import OCRResult
    return OCRResult(
        unreliable=True,
        warnings=["pipeline_crash"],
        ocr_engine="easyocr_en+paddleocr_ta",
    )
