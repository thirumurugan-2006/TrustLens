from app.input.schemas import NormalizedPost, Platform
from app.validation.schemas import ValidationResult, ValidationStatus
from app.validation.url_validator import validate_reddit_url

_ALLOWED_PLATFORMS = {p.value for p in Platform if p != Platform.unknown}

def validate_post(post: NormalizedPost | None) -> ValidationResult:
    errors = []
    if post is None:
        return ValidationResult(
            valid=False,
            status=ValidationStatus.INVALID,
            errors=["NormalizedPost is required."]
        )
    if not post.post_id:
        errors.append("post_id is required.")
        
    platform_val = post.platform.value if isinstance(post.platform, Platform) else str(post.platform or "").strip()
    if not platform_val:
        errors.append("platform is required.")
    elif platform_val == Platform.unknown.value or platform_val == "unknown":
        errors.append("Unknown or unsupported platform: unknown.")
    elif platform_val not in _ALLOWED_PLATFORMS:
        errors.append(f"Unsupported platform: {platform_val}.")
        
    if not post.timestamp:
        errors.append("timestamp is required.")
    if not isinstance(post.media.images, list):
        errors.append("images must be a list.")
    if not isinstance(post.comments, list):
        errors.append("comments must be a list.")
    if post.source_url is not None:
        url_result = validate_reddit_url(post.source_url)
        if not url_result.valid:
            errors.extend(url_result.errors)

    if errors:
        is_unsupported = any("unsupported" in e.lower() or "unknown" in e.lower() for e in errors)
        status = ValidationStatus.UNSUPPORTED if is_unsupported else ValidationStatus.INVALID
        return ValidationResult(valid=False, status=status, errors=errors)

    return ValidationResult(valid=True, status=ValidationStatus.VALID, errors=[])

