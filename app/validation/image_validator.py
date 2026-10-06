from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.validation.schemas import ValidationResult

SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200


def validate_image(image_path: str | Path) -> ValidationResult:
    path = Path(image_path)
    if not path.is_file():
        return ValidationResult(valid=False, errors=["Image file does not exist."])

    if path.stat().st_size > MAX_IMAGE_SIZE_BYTES:
        return ValidationResult(valid=False, errors=["Image file exceeds the 10 MB limit."])

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image_format = image.format
            width, height = image.size
    except (UnidentifiedImageError, OSError):
        return ValidationResult(valid=False, errors=["Invalid image file."])

    if image_format not in SUPPORTED_IMAGE_FORMATS:
        return ValidationResult(valid=False, errors=["Unsupported image format."])

    warnings = []
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        warnings.append("Image resolution is below the recommended minimum.")

    return ValidationResult(
        valid=True,
        warnings=warnings,
        quality_score=1.0 if not warnings else 0.75,
        quality_level="high" if not warnings else "medium",
    )


def image_metadata(image_path: str | Path) -> dict[str, int | str]:
    path = Path(image_path)
    with Image.open(path) as image:
        return {
            "image_format": image.format or "unknown",
            "image_width": image.width,
            "image_height": image.height,
            "image_size_bytes": path.stat().st_size,
        }
