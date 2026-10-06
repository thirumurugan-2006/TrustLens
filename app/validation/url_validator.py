from urllib.parse import urlparse
import re

from app.validation.schemas import ValidationResult

_REDDIT_POST_PATH = re.compile(r"^/r/[^/]+/comments/[A-Za-z0-9]+(?:/[^/]*)?/?$")


def validate_reddit_url(url: str | None) -> ValidationResult:
    if not url:
        return ValidationResult(valid=False, errors=["Invalid Reddit post URL."])

    try:
        parsed = urlparse(url)
    except ValueError:
        return ValidationResult(valid=False, errors=["Invalid Reddit post URL."])

    hostname = (parsed.hostname or "").lower()
    valid = (
        parsed.scheme == "https"
        and hostname in {"reddit.com", "www.reddit.com"}
        and bool(_REDDIT_POST_PATH.match(parsed.path))
    )
    if not valid:
        return ValidationResult(valid=False, errors=["Invalid Reddit post URL."])

    return ValidationResult(valid=True, quality_score=1.0, quality_level="high")

def validate_platform_url(url: str | None) -> ValidationResult:
    if not url:
        return ValidationResult(valid=False, errors=["Invalid URL."])

    try:
        parsed = urlparse(url)
    except ValueError:
        return ValidationResult(valid=False, errors=["Invalid URL."])

    if parsed.scheme not in ("http", "https"):
        return ValidationResult(valid=False, errors=["URL must use HTTP/HTTPS."])

    if not parsed.netloc:
        return ValidationResult(valid=False, errors=["URL must contain a valid domain."])

    return ValidationResult(valid=True, quality_score=1.0, quality_level="high")

