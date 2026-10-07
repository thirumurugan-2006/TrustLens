import pytest
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image

from app.input.schemas import (
    NormalizedPost,
    Platform,
    PostType,
    PostContent,
    PostMedia,
)
from app.validation.schemas import ValidationResult, ValidationStatus
from app.validation.post_validator import validate_post
from app.validation.text_validator import validate_text
from app.validation.image_validator import validate_image
from app.pipeline.input_pipeline import InputPipeline


def test_unknown_platform_rejected():
    """L2 Hardening: Platform.unknown or 'unknown' must be rejected as invalid/unsupported."""
    post = NormalizedPost(
        post_id="post_unknown_01",
        platform=Platform.unknown,
        post_type=PostType.text,
        content=PostContent(text="This should be rejected."),
        timestamp="2026-10-07T00:00:00+00:00",
    )
    result = validate_post(post)
    assert not result.valid
    assert result.status == ValidationStatus.UNSUPPORTED
    assert any("unknown" in err.lower() or "unsupported" in err.lower() for err in result.errors)

    post_str = post.model_copy(update={"platform": "unknown"})
    result_str = validate_post(post_str)
    assert not result_str.valid
    assert result_str.status == ValidationStatus.UNSUPPORTED


def test_unsupported_platform_rejected():
    """L2 Hardening: Any unregistered or unsupported platform string must be rejected."""
    post = NormalizedPost(
        post_id="post_unsupp_01",
        platform="nonexistent_social_platform",
        post_type=PostType.text,
        content=PostContent(text="Unsupported platform test."),
        timestamp="2026-10-07T00:00:00+00:00",
    )
    result = validate_post(post)
    assert not result.valid
    assert result.status == ValidationStatus.UNSUPPORTED
    assert any("unsupported platform" in err.lower() for err in result.errors)


def test_valid_platforms_accepted():
    """L2 Hardening: Registered platforms (reddit, generic, text, screenshot, etc.) pass."""
    for p in [Platform.generic, Platform.text, Platform.screenshot, Platform.reddit, Platform.youtube]:
        post = NormalizedPost(
            post_id=f"post_{p.value}",
            platform=p,
            post_type=PostType.text,
            content=PostContent(text="Legitimate platform post."),
            timestamp="2026-10-07T00:00:00+00:00",
        )
        result = validate_post(post)
        assert result.valid, f"Platform {p} failed validation: {result.errors}"
        assert result.status == ValidationStatus.VALID


def test_empty_and_whitespace_text():
    """L2 Hardening: Empty or whitespace-only text is rejected."""
    assert not validate_text(None).valid
    assert not validate_text("").valid
    assert not validate_text("    ").valid
    assert not validate_text("\n\t  \r\n").valid
    res = validate_text("   ")
    assert res.status == ValidationStatus.INSUFFICIENT


def test_multilingual_text_validation():
    """L2 Hardening: Deterministic text validation across diverse scripts and symbols."""
    # Tamil
    assert validate_text("இந்த முதலீடு மிக அதிக லாபம் தரும்").valid
    # Hindi
    assert validate_text("यह योजना आपको प्रति माह ₹50,000 देगी").valid
    # Tanglish
    assert validate_text("Intha scheme la invest panna nalla return kedaikum").valid
    # Financial symbols & URLs & emojis
    assert validate_text("Earn ₹50,000 / $1,000 daily! Visit https://example.com 🚀💸").valid


def test_screenshot_validation(tmp_path: Path):
    """L2 Hardening: Valid and corrupt screenshot validation."""
    valid_img = tmp_path / "valid.png"
    Image.new("RGB", (300, 300), "white").save(valid_img)
    valid_res = validate_image(valid_img)
    assert valid_res.valid
    assert valid_res.status in (ValidationStatus.VALID, ValidationStatus.LOW_QUALITY)

    corrupt_img = tmp_path / "corrupt.png"
    corrupt_img.write_bytes(b"not a valid png binary stream")
    corrupt_res = validate_image(corrupt_img)
    assert not corrupt_res.valid
    assert corrupt_res.status == ValidationStatus.INVALID


def test_malformed_post():
    """L2 Hardening: Missing required fields (post_id, timestamp) fail post validation."""
    post_no_id = NormalizedPost(
        post_id="",
        platform=Platform.generic,
        post_type=PostType.text,
        content=PostContent(text="Text"),
        timestamp="2026-10-07T00:00:00+00:00",
    )
    res = validate_post(post_no_id)
    assert not res.valid
    assert res.status == ValidationStatus.INVALID
    assert "post_id is required." in res.errors

    post_no_ts = NormalizedPost(
        post_id="p123",
        platform=Platform.generic,
        post_type=PostType.text,
        content=PostContent(text="Text"),
        timestamp=None,
    )
    res_ts = validate_post(post_no_ts)
    assert not res_ts.valid
    assert "timestamp is required." in res_ts.errors


def test_none_post_validation():
    """L2 Hardening: None post is rejected."""
    res = validate_post(None)
    assert not res.valid
    assert res.status == ValidationStatus.INVALID
    assert "NormalizedPost is required." in res.errors


def test_valid_text_and_image_post(tmp_path: Path):
    """L2 Hardening: Multi-modal post with text and image passes."""
    img_path = tmp_path / "photo.jpg"
    Image.new("RGB", (250, 250), "blue").save(img_path)

    post = NormalizedPost(
        post_id="post_multi_01",
        platform=Platform.generic,
        post_type=PostType.image_text,
        content=PostContent(text="Check this investment screenshot"),
        media=PostMedia(images=[str(img_path)]),
        timestamp="2026-10-07T00:00:00+00:00",
    )
    res = validate_post(post)
    assert res.valid
    assert res.status == ValidationStatus.VALID


def test_invalid_input_blocked_from_semantic_processing():
    """L2 Hardening: Invalid input is rejected before semantic pipeline / claim extraction."""
    pipeline = InputPipeline()
    pipeline.claim_extractor = MagicMock()
    pipeline.claim_decomposer = MagicMock()

    # Empty text must raise ValueError and never call claim_extractor
    with pytest.raises(ValueError):
        pipeline.process("text", "   ")
    pipeline.claim_extractor.extract_from_post.assert_not_called()

    # Corrupt screenshot must raise ValueError and never call claim_extractor
    with pytest.raises(ValueError):
        pipeline.process("screenshot", "nonexistent_file.png")
    pipeline.claim_extractor.extract_from_post.assert_not_called()
