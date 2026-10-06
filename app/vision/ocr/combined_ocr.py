"""
TrustLens Step 11 — Combined OCR Architecture
==============================================

Implements the CombinedOCR class that routes images through:
  - Tamil backend (PaddleOCR) for Tamil/unknown content
  - English backend (EasyOCR) for English content

Language detection uses script evidence from the image, NOT raw confidence.
Tamil OCR failures are never automatically relabelled as English.

Architecture::

              IMAGE
                ↓
          Text Detection
                ↓
           Text Regions
                ↓
      ┌─────────┴─────────┐
      ↓                   ↓
 Tamil Reader        English Reader
 PaddleOCR           EasyOCR (en)
      ↓                   ↓
      └─────────┬─────────┘
                ↓
          Result Fusion
                ↓
          Language Label
                ↓
         Reliability Score
                ↓
          Final OCR Text
"""
from __future__ import annotations

import time
from pathlib import Path

from app.vision.ocr.ocr_result import TamilOCRResult, OCR_STATUS_FAILED
from app.vision.ocr.tamil_script import (
    detect_language_label,
    tamil_ratio,
    valid_char_ratio,
)
from app.vision.ocr.tamil_reliability import (
    compute_reliability_score,
    is_reliable,
    ocr_status_from_result,
)


# ---------------------------------------------------------------------------
# EasyOCR English Backend
# ---------------------------------------------------------------------------
class EasyOCREnglishBackend:
    """
    EasyOCR English reader wrapper.
    The Tamil EasyOCR checkpoint has a vocabulary mismatch and cannot be loaded.
    Only the English backend is used from EasyOCR.
    """

    name = "easyocr_en"

    def __init__(self, model_directory: str | Path = "data/models/easyocr") -> None:
        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        self._reader = None

    def _load(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                ["en"],
                gpu=False,
                model_storage_directory=str(self.model_directory),
                verbose=False,
            )
        return self._reader

    def read(self, image_path: str | Path) -> tuple[str, float | None, list[float], list, int, list[str]]:
        """
        Returns (text, avg_confidence, region_confidences, bboxes, detection_count, region_texts).
        """
        results = self._load().readtext(str(image_path), detail=1)
        if not results:
            return "", None, [], [], 0
        texts = [r[1] for r in results]
        confs = [float(r[2]) for r in results]
        bboxes = [r[0] for r in results]
        text = " ".join(texts)
        avg_conf = sum(confs) / len(confs)
        return text, avg_conf, confs, bboxes, len(results), texts


# ---------------------------------------------------------------------------
# PaddleOCR Tamil Backend
# ---------------------------------------------------------------------------
class PaddleOCRTamilBackend:
    """
    PaddleOCR Tamil reader wrapper using ta_PP-OCRv5_mobile_rec.
    This is the currently working Tamil OCR backend in TrustLens.
    """

    name = "paddleocr_ta"

    def __init__(self) -> None:
        self._engine = None
        self.error: str = ""

    def _load(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
                self._engine = PaddleOCR(
                    lang="ta",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except Exception as exc:
                self.error = str(exc)
                raise RuntimeError(f"PaddleOCR Tamil unavailable: {exc}") from exc
        return self._engine

    def read(self, image_path: str | Path) -> tuple[str, float | None, list[float], list, int, list[str]]:
        """
        Returns (text, avg_confidence, region_confidences, bboxes, detection_count, region_texts).
        """
        try:
            results = self._load().predict(str(image_path))
        except Exception as exc:
            raise RuntimeError(f"PaddleOCR Tamil inference failed: {exc}") from exc

        texts, confs, bboxes = [], [], []
        for page in results or []:
            if not hasattr(page, "get"):
                continue
            page_texts = page.get("rec_texts", [])
            page_scores = page.get("rec_scores", [])
            page_boxes = page.get("rec_boxes", [])
            for i, t in enumerate(page_texts):
                texts.append(t)
                confs.append(float(page_scores[i]) if i < len(page_scores) else 0.0)
                bboxes.append(page_boxes[i] if i < len(page_boxes) else None)

        if not texts:
            return "", None, [], [], 0, []
        text = " ".join(texts)
        avg_conf = sum(confs) / len(confs)
        return text, avg_conf, confs, bboxes, len(texts), texts


# ---------------------------------------------------------------------------
# CombinedOCR — the main TrustLens pipeline reader
# ---------------------------------------------------------------------------
class CombinedOCR:
    """
    Combined OCR reader: routes through Tamil and English backends, fuses
    results, assigns a language label, and computes a calibrated reliability
    score.

    Language assignment follows script evidence, not raw confidence.
    Tamil OCR failures are never silently relabelled as English.
    """

    def __init__(
        self,
        model_directory: str | Path = "data/models/easyocr",
    ) -> None:
        self._tamil = PaddleOCRTamilBackend()
        self._english = EasyOCREnglishBackend(model_directory)
        self._tamil_unavailable = False

    def read(self, image_path: str | Path) -> TamilOCRResult:
        """
        Run combined OCR on an image.

        Decision logic:
          1. Run English reader (always available).
          2. If the image has Tamil script evidence, also run Tamil reader.
          3. Select the result with higher selection score.
          4. Assign language from script evidence.
          5. Compute TrustLens reliability score.
          6. Assign OCR status.

        Tamil failures are reported as status=OCR_FAILED with
        language="unknown" — never silently converted to language="en".
        """
        started = time.perf_counter()
        image_path = Path(image_path)

        # ── English pass ────────────────────────────────────────────────────
        try:
            en_text, en_conf, en_region_confs, en_bboxes, en_count, en_region_texts = self._english.read(image_path)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            return TamilOCRResult(
                error=f"English OCR failed: {exc}",
                status=OCR_STATUS_FAILED,
                backend=self._english.name,
                processing_time_ms=elapsed,
            )

        en_ta_ratio = tamil_ratio(en_text)
        en_vc_ratio = valid_char_ratio(en_text)

        # ── Decide whether to run Tamil reader ───────────────────────────────
        # If English output contains Tamil script, the image probably has Tamil.
        # Also try Tamil when English produces very little usable text.
        run_tamil = (en_ta_ratio > 0.0) or (en_count == 0) or (en_vc_ratio < 0.3)

        ta_text, ta_conf, ta_region_confs, ta_bboxes, ta_count, ta_region_texts = "", None, [], [], 0, []
        tamil_error = ""

        if run_tamil and not self._tamil_unavailable:
            try:
                ta_text, ta_conf, ta_region_confs, ta_bboxes, ta_count, ta_region_texts = self._tamil.read(image_path)
            except RuntimeError as exc:
                self._tamil_unavailable = True
                tamil_error = str(exc)

        # ── Compute per-candidate selection scores ───────────────────────────
        def _selection_score(text, avg_conf, region_confs, expected_lang):
            conf = float(avg_conf) if avg_conf is not None else 0.0
            ta_r = tamil_ratio(text)
            vc_r = valid_char_ratio(text)
            length_norm = min(len(text.strip()) / 50.0, 1.0)
            script_score = ta_r if expected_lang == "ta" else (1.0 - ta_r)
            return 0.40 * conf + 0.25 * script_score + 0.20 * vc_r + 0.15 * length_norm

        en_score = _selection_score(en_text, en_conf, en_region_confs, "en")
        ta_score = _selection_score(ta_text, ta_conf, ta_region_confs, "ta") if ta_text else -1.0

        # ── Select best candidate ────────────────────────────────────────────
        if ta_text and ta_score > en_score:
            text = ta_text
            raw_conf = ta_conf
            region_confs = ta_region_confs
            bboxes = ta_bboxes
            detection_count = ta_count
            backend = self._tamil.name
            region_texts = ta_region_texts
        else:
            text = en_text
            raw_conf = en_conf
            region_confs = en_region_confs
            bboxes = en_bboxes
            detection_count = en_count
            backend = self._english.name
            region_texts = en_region_texts
            
        # ── Normalization: region sorting and deduplication ──────────────────
        from app.preprocessing.normalization.text_normalizer import _sort_regions, _remove_duplicates
        
        region_texts, bboxes, region_confs = _sort_regions(region_texts, bboxes, region_confs)
        region_texts, bboxes, region_confs = _remove_duplicates(region_texts, bboxes, region_confs)
        
        # Re-join texts after sorting and deduplicating
        if region_texts:
            text = " ".join(region_texts)
        else:
            text = ""

        # ── Language detection from script evidence ──────────────────────────
        # NEVER assign language="en" because Tamil confidence is low.
        language = detect_language_label(text, raw_conf)

        # ── Reliability score (TrustLens calibrated) ─────────────────────────
        expected_lang_for_scoring = language if language in ("ta", "en") else "ta"
        reliability = compute_reliability_score(
            text=text,
            raw_confidence=raw_conf,
            region_confidences=region_confs,
            backend=backend,
            expected_language=expected_lang_for_scoring,
        )

        # ── OCR status ───────────────────────────────────────────────────────
        status = ocr_status_from_result(text, raw_conf, reliability, tamil_error)
        if language == "unknown":
            status = "LANGUAGE_UNCERTAIN"

        elapsed = (time.perf_counter() - started) * 1000

        result = TamilOCRResult(
            text=text,
            raw_text=text,
            language=language,
            raw_confidence=raw_conf,
            reliability_score=reliability,
            bbox=bboxes,
            backend=backend,
            status=status,
            tamil_ratio=tamil_ratio(text),
            valid_char_ratio=valid_char_ratio(text),
            detection_count=detection_count,
            region_confidences=region_confs,
            reliable=is_reliable(reliability),
            error=tamil_error,
            processing_time_ms=elapsed,
        )
        
        from app.preprocessing.normalization.text_normalizer import normalize_ocr_result
        return normalize_ocr_result(result)
