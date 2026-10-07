from pathlib import Path

from PIL import Image

from app.input.schemas import NormalizedPost, Platform, PostType, PostContent
from app.validation.image_validator import validate_image
from app.validation.post_validator import validate_post
from app.validation.quality_checker import check_ocr_quality
from app.validation.text_validator import validate_text
from app.validation.url_validator import validate_reddit_url


def test_text_validation():
    assert not validate_text(None).valid
    assert not validate_text("   ").valid
    assert validate_text("Scam?").valid
    assert validate_text("Scam?").warnings
    assert validate_text("This is usable text.").valid


def test_image_validation(tmp_path: Path):
    missing = validate_image(tmp_path / "missing.png")
    assert not missing.valid

    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image", encoding="utf-8")
    assert not validate_image(invalid).valid

    unsupported = tmp_path / "unsupported.bmp"
    Image.new("RGB", (300, 300), "white").save(unsupported)
    unsupported_result = validate_image(unsupported)
    assert not unsupported_result.valid
    assert "Unsupported image format." in unsupported_result.errors

    small = tmp_path / "small.png"
    Image.new("RGB", (50, 50), "white").save(small)
    small_result = validate_image(small)
    assert small_result.valid
    assert small_result.warnings

    valid = tmp_path / "valid.png"
    Image.new("RGB", (300, 300), "white").save(valid)
    assert validate_image(valid).valid

    oversized = tmp_path / "oversized.png"
    oversized.write_bytes(b"0" * (10 * 1024 * 1024 + 1))
    assert not validate_image(oversized).valid


def test_ocr_quality_levels():
    assert check_ocr_quality("text", 0.90).quality_level == "high"
    medium = check_ocr_quality("text", 0.73)
    assert medium.quality_level == "medium"
    assert medium.warnings
    low = check_ocr_quality("text", 0.40)
    assert low.quality_level == "low"
    assert low.warnings
    failed = check_ocr_quality("", 0.90)
    assert failed.quality_level == "failed"
    assert failed.warnings


def test_reddit_url_validation():
    valid = "https://www.reddit.com/r/example/comments/abc123/example/"
    assert validate_reddit_url(valid).valid
    for invalid in ["hello", "example.com", "ftp://reddit.com/x", "https://google.com"]:
        assert not validate_reddit_url(invalid).valid


def test_normalized_post_validation():
    post = NormalizedPost(
        post_id="text_input",
        platform=Platform.generic,
        post_type=PostType.text,
        content=PostContent(text="usable text"),
        timestamp="2026-10-05T00:00:00+00:00",
    )
    assert validate_post(post).valid

    invalid = post.model_copy(update={"platform": "unknown"})
    assert not validate_post(invalid).valid
