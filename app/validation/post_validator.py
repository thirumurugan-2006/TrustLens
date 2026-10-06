from app.input.schemas import NormalizedPost
from app.validation.schemas import ValidationResult
from app.validation.url_validator import validate_reddit_url

_ALLOWED_PLATFORMS = {"text", "screenshot", "reddit"}


def validate_post(post: NormalizedPost | None) -> ValidationResult:
    errors = []
    if post is None:
        return ValidationResult(valid=False, errors=["NormalizedPost is required."])
    if not post.post_id:
        errors.append("post_id is required.")
    if not post.platform:
        errors.append("platform is required.")
    elif post.platform not in _ALLOWED_PLATFORMS:
        errors.append("Unsupported platform.")
    if not post.timestamp:
        errors.append("timestamp is required.")
    if not isinstance(post.images, list):
        errors.append("images must be a list.")
    if not isinstance(post.comments, list):
        errors.append("comments must be a list.")
    if post.source_url is not None:
        url_result = validate_reddit_url(post.source_url)
        if not url_result.valid:
            errors.extend(url_result.errors)

    return ValidationResult(valid=not errors, errors=errors)
