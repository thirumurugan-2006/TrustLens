from app.validation.schemas import ValidationResult

MIN_TEXT_LENGTH = 10


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def validate_text(text: str | None) -> ValidationResult:
    if text is None:
        return ValidationResult(valid=False, errors=["Text input cannot be empty."])

    normalized_text = normalize_text(text)
    if not normalized_text:
        return ValidationResult(valid=False, errors=["Text input cannot be empty."])

    warnings = []
    if len(normalized_text) < MIN_TEXT_LENGTH:
        warnings.append("Text input is very short.")

    return ValidationResult(
        valid=True,
        warnings=warnings,
        quality_score=1.0 if not warnings else 0.75,
        quality_level="high" if not warnings else "medium",
    )
