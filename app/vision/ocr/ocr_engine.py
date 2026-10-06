"""
TrustLens OCR v2 — OCR Engine Service
=======================================

Provides lazy-loaded, singleton OCR reader instances for:
  - English:  EasyOCR(["en"])
  - Tamil:    PaddleOCR(lang="ta")

Critical design decisions
--------------------------
1. easyocr.Reader(["en", "ta"]) is NEVER used.
   The EasyOCR Tamil checkpoint has a vocabulary mismatch:
       checkpoint  Prediction.weight = [143, 512]
       model       Prediction.weight = [127, 512]
   This raises RuntimeError at load time.

2. Tamil OCR uses PaddleOCR (ta_PP-OCRv5_mobile_rec) which works correctly.

3. Readers are lazy-loaded once per process lifetime (singleton pattern).
   They are NOT re-initialized per request.

4. If either reader fails to load, the error is captured and exposed via
   warnings — the API continues with the remaining reader.

5. Model files are stored under data/models/easyocr/ (configurable).
   Absolute paths are never hard-coded.

Reader interface
----------------
Both backends expose:
    read(image_path) → tuple(text, avg_confidence, region_confs, bboxes, count, region_texts)

The OCREngine.run_ocr() method returns a raw result dict compatible with
the quality evaluator and candidate selector.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# EasyOCR English backend
# ---------------------------------------------------------------------------

class _EasyOCREnglishBackend:
    """
    Lazy-loaded EasyOCR English reader.

    easyocr.Reader(["en"]) works correctly in this environment.
    easyocr.Reader(["en", "ta"]) is NEVER used here — checkpoint mismatch.
    """

    name = "easyocr_en"

    def __init__(self, model_directory: Path) -> None:
        self._model_dir = model_directory
        self._reader = None
        self._error: str = ""

    def _load(self):
        if self._reader is None and not self._error:
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    ["en"],
                    gpu=False,
                    model_storage_directory=str(self._model_dir),
                    verbose=False,
                )
                logger.info("EasyOCR English reader loaded.")
            except Exception as exc:
                self._error = str(exc)
                logger.error("EasyOCR English reader failed to load: %s", exc)
        return self._reader

    def is_available(self) -> bool:
        return self._load() is not None

    def error_message(self) -> str:
        return self._error

    def read(
        self, image_path: str | Path
    ) -> tuple[str, Optional[float], list[float], list, int, list[str]]:
        """
        Returns (text, avg_confidence, region_confs, bboxes, count, region_texts).
        """
        reader = self._load()
        if reader is None:
            return "", None, [], [], 0, []

        try:
            results = reader.readtext(str(image_path), detail=1)
        except Exception as exc:
            logger.warning("EasyOCR English read failed: %s", exc)
            return "", None, [], [], 0, []

        if not results:
            return "", None, [], [], 0, []

        texts  = [r[1] for r in results]
        confs  = [float(r[2]) for r in results]
        bboxes = [r[0] for r in results]
        text   = " ".join(texts)
        avg    = sum(confs) / len(confs)
        return text, avg, confs, bboxes, len(results), texts


# ---------------------------------------------------------------------------
# PaddleOCR Tamil backend
# ---------------------------------------------------------------------------

class _PaddleOCRTamilBackend:
    """
    Lazy-loaded PaddleOCR Tamil reader.

    Uses lang="ta" which maps to ta_PP-OCRv5_mobile_rec.
    This is the working Tamil OCR backend in TrustLens because the
    EasyOCR Tamil checkpoint has a vocabulary mismatch.
    """

    name = "paddleocr_ta"

    def __init__(self) -> None:
        self._engine = None
        self._error: str = ""

    def _load(self):
        if self._engine is None and not self._error:
            try:
                from paddleocr import PaddleOCR
                self._engine = PaddleOCR(
                    lang="ta",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                )
                logger.info("PaddleOCR Tamil reader loaded.")
            except Exception as exc:
                self._error = str(exc)
                logger.error("PaddleOCR Tamil reader failed to load: %s", exc)
        return self._engine

    def is_available(self) -> bool:
        return self._load() is not None

    def error_message(self) -> str:
        return self._error

    def read(
        self, image_path: str | Path
    ) -> tuple[str, Optional[float], list[float], list, int, list[str]]:
        """
        Returns (text, avg_confidence, region_confs, bboxes, count, region_texts).
        """
        engine = self._load()
        if engine is None:
            return "", None, [], [], 0, []

        try:
            results = engine.predict(str(image_path))
        except Exception as exc:
            logger.warning("PaddleOCR Tamil read failed: %s", exc)
            return "", None, [], [], 0, []

        texts, confs, bboxes = [], [], []
        for page in results or []:
            if not hasattr(page, "get"):
                continue
            page_texts  = page.get("rec_texts", [])
            page_scores = page.get("rec_scores", [])
            page_boxes  = page.get("rec_boxes", [])
            for i, t in enumerate(page_texts):
                texts.append(t)
                confs.append(float(page_scores[i]) if i < len(page_scores) else 0.0)
                bboxes.append(page_boxes[i] if i < len(page_boxes) else None)

        if not texts:
            return "", None, [], [], 0, []

        text = " ".join(texts)
        avg  = sum(confs) / len(confs)
        return text, avg, confs, bboxes, len(texts), texts


# ---------------------------------------------------------------------------
# OCR Engine Service — public API
# ---------------------------------------------------------------------------

class OCRService:
    """
    Application-level OCR service with lazy-loaded singleton readers.

    Maintains one English reader and one Tamil reader for the entire
    process lifetime.  Never recreates readers per request.

    Usage
    -----
    service = OCRService()              # or OCRService(model_directory=...)
    result  = service.run_ocr(image_path, language_mode=["en"])
    result  = service.run_ocr(image_path, language_mode=["ta"])
    """

    def __init__(
        self,
        model_directory: str | Path = "data/models/easyocr",
    ) -> None:
        model_dir = Path(model_directory)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        self._model_dir = model_dir
        self._english   = _EasyOCREnglishBackend(model_dir)
        self._tamil     = _PaddleOCRTamilBackend()
        self.preprocessor = None  # set by OCRPipeline if needed

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------

    def english_available(self) -> bool:
        return self._english.is_available()

    def tamil_available(self) -> bool:
        return self._tamil.is_available()

    def english_error(self) -> str:
        return self._english.error_message()

    def tamil_error(self) -> str:
        return self._tamil.error_message()

    # ------------------------------------------------------------------
    # Raw read
    # ------------------------------------------------------------------

    def run_ocr(
        self,
        image_path: str | Path,
        language_mode: list[str],
        preprocessing: str = "original",
    ) -> dict:
        """
        Run OCR for the given language mode on an (already pre-processed) image.

        Returns a raw candidate dict ready for evaluate_candidate().

        Parameters
        ----------
        image_path : str | Path
            Path to the (pre-processed) image file.
        language_mode : list[str]
            ["en"] or ["ta"].
        preprocessing : str
            Name label for the preprocessing variant (for metadata only).

        Returns
        -------
        dict with keys:
            text, confidence, language_mode, preprocessing, detection_count,
            backend, warnings
        """
        if language_mode == ["en"]:
            text, avg, region_confs, bboxes, count, region_texts = self._english.read(image_path)
            backend = self._english.name
            warnings: list[str] = []
            if not self._english.is_available():
                warnings.append("english_ocr_unavailable")
        elif language_mode == ["ta"]:
            text, avg, region_confs, bboxes, count, region_texts = self._tamil.read(image_path)
            backend = self._tamil.name
            warnings = []
            if not self._tamil.is_available():
                warnings.append("tamil_ocr_unavailable")
        else:
            raise ValueError(f"Unsupported language_mode: {language_mode}")

        return {
            "text": text or "",
            "confidence": float(avg) if avg is not None else 0.0,
            "language_mode": language_mode,
            "preprocessing": preprocessing,
            "detection_count": count,
            "backend": backend,
            "region_confidences": region_confs,
            "_bboxes": bboxes,
            "_region_texts": region_texts,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Legacy compatibility helpers (used by existing tests/monkeypatching)
    # ------------------------------------------------------------------

    def english_reader(self):
        """Return the raw EasyOCR reader (legacy interface for tests)."""
        return self._english._load()

    def combined_reader(self):
        """
        Compatibility shim — returns a CombinedOCR wrapper object
        used by existing code paths.
        """
        from app.vision.ocr.combined_ocr import CombinedOCR
        return CombinedOCR(self._model_dir)
