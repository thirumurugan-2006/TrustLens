from app.validation.schemas import ValidationResult

HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.60


def check_ocr_quality(ocr_text: str | None, ocr_confidence: float) -> ValidationResult:
    if not ocr_text or not ocr_text.strip():
        return ValidationResult(
            valid=True,
            warnings=["OCR did not extract usable text."],
            quality_score=0.0,
            quality_level="failed",
        )

    if ocr_confidence >= HIGH_CONFIDENCE:
        return ValidationResult(
            valid=True,
            quality_score=ocr_confidence,
            quality_level="high",
        )
    if ocr_confidence >= MEDIUM_CONFIDENCE:
        return ValidationResult(
            valid=True,
            warnings=["OCR confidence is moderate."],
            quality_score=ocr_confidence,
            quality_level="medium",
        )
    return ValidationResult(
        valid=True,
        warnings=["OCR confidence is low."],
        quality_score=ocr_confidence,
        quality_level="low",
    )
