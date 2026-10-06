#!/usr/bin/env python3
"""
TrustLens OCR v2 — Comprehensive Test Script
==============================================

Tests the OCR v2 pipeline against multiple image types:
  1. English screenshot
  2. Tamil screenshot
  3. Tamil + English mixed screenshot
  4. Very low quality / blank screenshot
  5. Invalid image (non-image file)

Usage
-----
From the TrustLens project root:

    python scripts/test_ocr_v2.py

Or with specific images:

    python scripts/test_ocr_v2.py path/to/image1.png path/to/image2.png

Output format:
    A readable report with candidate table and selection summary.

Note
----
This script requires the OCR models to be available:
  - EasyOCR English: data/models/easyocr/
  - PaddleOCR Tamil: downloads automatically on first use

The script does NOT force any result — it reports what the pipeline
actually produces. If Tamil OCR is poor, it reports the actual numbers.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Image helpers — create synthetic test images using PIL
# ---------------------------------------------------------------------------

def _create_english_image(path: Path) -> Path:
    """Create a synthetic English investment screenshot."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (600, 300), "white")
        draw = ImageDraw.Draw(img)
        lines = [
            "Investment Growth Plan 2024",
            "Monthly Return: 15% guaranteed",
            "Minimum Investment: $500",
            "Contact: investor@example.com",
            "Register now and earn big!",
        ]
        y = 20
        for line in lines:
            draw.text((20, y), line, fill="black")
            y += 45
        img.save(path)
        return path
    except Exception as e:
        print(f"  [WARNING] Could not create English image: {e}")
        return path


def _create_tamil_image(path: Path) -> Path:
    """
    Create a synthetic Tamil investment screenshot.
    Uses pre-rendered Tamil Unicode text embedded with PIL.
    Note: Actual Tamil rendering requires a Tamil font installed.
    Falls back to a simple image if font not available.
    """
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(img)
        # Use Tamil Unicode directly — rendering quality depends on PIL + font
        lines = [
            "முதலீட்டு திட்டம் 2024",
            "மாத வருமானம்: 15%",
            "குறைந்தபட்ச முதலீடு: ₹500",
            "தொடர்பு: info@example.com",
            "இப்போதே பதிவு செய்யுங்கள்",
        ]
        y = 20
        for line in lines:
            draw.text((20, y), line, fill="black")
            y += 60
        img.save(path)
        return path
    except Exception as e:
        print(f"  [WARNING] Could not create Tamil image: {e}")
        return path


def _create_mixed_image(path: Path) -> Path:
    """Create a synthetic Tamil + English mixed screenshot."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(img)
        lines = [
            "இந்த job மிகவும் நல்லது",
            "Apply pannunga today",
            "மாதம் ₹50,000 salary",
            "Work from home opportunity",
            "Guaranteed income monthly",
        ]
        y = 20
        for line in lines:
            draw.text((20, y), line, fill="black")
            y += 60
        img.save(path)
        return path
    except Exception as e:
        print(f"  [WARNING] Could not create mixed image: {e}")
        return path


def _create_blank_image(path: Path) -> Path:
    """Create a very low quality blank image."""
    try:
        from PIL import Image
        img = Image.new("RGB", (100, 60), "white")
        img.save(path)
        return path
    except Exception as e:
        print(f"  [WARNING] Could not create blank image: {e}")
        return path


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

SEP = "=" * 60
DASH = "-" * 60


def _print_candidate_table(candidates: list[dict]) -> None:
    if not candidates:
        print("  (no candidates)")
        return

    header = (
        f"  {'Lang':<8} {'Preprocess':<22} {'Conf':>6} {'Tamil':>6} "
        f"{'Latin':>6} {'Score':>6} {'Usable':<8} {'Backend'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for c in candidates:
        lang   = str(c.get("language_mode", [])).replace("'", "")
        prep   = str(c.get("preprocessing", "?"))[:20]
        conf   = c.get("confidence", 0.0)
        ta     = c.get("tamil_ratio", 0.0)
        la     = c.get("latin_ratio", 0.0)
        score  = c.get("selection_score", 0.0)
        usable = "YES" if c.get("usable") else "NO"
        backend = c.get("backend", "?")[:20]
        print(
            f"  {lang:<8} {prep:<22} {conf:>6.3f} {ta:>6.3f} "
            f"{la:>6.3f} {score:>6.3f} {usable:<8} {backend}"
        )


def _print_selected(metadata: dict, text: str) -> None:
    print("\nSELECTED:")
    print(f"  Language mode  : {metadata.get('ocr_selected_language_mode')}")
    print(f"  Preprocessing  : {metadata.get('ocr_selected_preprocessing', 'unknown')}")
    print(f"  Backend        : {metadata.get('ocr_engine', 'unknown')}")
    print(f"  Confidence     : {metadata.get('ocr_confidence', 0.0):.4f}")
    print(f"  Selection score: {metadata.get('ocr_selection_score', 0.0):.4f}")
    print(f"  Tamil ratio    : {metadata.get('ocr_script_distribution', {}).get('Tamil', 0.0):.4f}")
    print(f"  Latin ratio    : {metadata.get('ocr_script_distribution', {}).get('Latin', 0.0):.4f}")
    print(f"  OCR quality    : {metadata.get('ocr_quality', '?')}")
    print(f"  OCR reliable   : {not metadata.get('ocr_unreliable', True)}")
    print(f"  Attempts       : {metadata.get('ocr_attempts', 0)}")
    print(f"  Detections     : {metadata.get('ocr_detection_count', 0)}")
    print(f"  Text length    : {metadata.get('ocr_text_length', 0)}")

    lang_analysis = metadata.get("language_analysis", {})
    print(f"  Primary lang   : {lang_analysis.get('primary_language', '?')}")
    print(f"  Lang reliable  : {lang_analysis.get('language_analysis_reliable', False)}")

    if text and text.strip():
        preview = text.strip()[:120].replace("\n", " ")
        print(f"\n  Text preview   : {preview}{'...' if len(text.strip()) > 120 else ''}")
    else:
        print("\n  Text preview   : (empty)")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(label: str, image_path: Path) -> None:
    print(f"\n{SEP}")
    print(f"TEST: {label}")
    print(f"IMAGE: {image_path}")
    print(SEP)

    if not image_path.exists():
        print(f"[SKIP] Image does not exist: {image_path}")
        return

    from app.input.loaders.screenshot_loader import ScreenshotParser

    try:
        parser = ScreenshotParser()
        post = parser.parse(image_path)
    except FileNotFoundError as exc:
        print(f"[SKIP] {exc}")
        return
    except ValueError as exc:
        print(f"[EXPECTED for invalid images] ValueError: {exc}")
        return
    except Exception as exc:
        print(f"[ERROR] Unexpected: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return

    metadata = post.metadata
    candidates = metadata.get("ocr_candidates", [])

    print("\nCANDIDATES:")
    _print_candidate_table(candidates)
    _print_selected(metadata, post.text or "")

    print()


def run_invalid_image_test(label: str, image_path: Path) -> None:
    print(f"\n{SEP}")
    print(f"TEST: {label}")
    print(f"IMAGE: {image_path}")
    print(SEP)

    from app.input.loaders.screenshot_loader import ScreenshotParser
    try:
        parser = ScreenshotParser()
        parser.parse(image_path)
        print("[FAIL] Expected ValueError for invalid image — none raised")
    except ValueError as exc:
        print(f"[PASS] ValueError raised as expected: {exc}")
    except FileNotFoundError as exc:
        print(f"[PASS] FileNotFoundError raised as expected: {exc}")
    except Exception as exc:
        print(f"[UNEXPECTED] {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEP)
    print("TrustLens OCR v2 — Test Report")
    print(SEP)

    # ── Check if specific images were passed as arguments ────────────────
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            run_test(f"User-provided: {arg}", Path(arg))
        return

    # ── Use provided test images or generate synthetic ones ──────────────
    test_img_dir = PROJECT_ROOT / "data" / "ocr_test" / "images"
    tmpdir = tempfile.mkdtemp()

    # --- Test 1: English screenshot ---
    en_candidates = sorted(test_img_dir.glob("english_baseline_*.png")) if test_img_dir.exists() else []
    if en_candidates:
        en_image = en_candidates[0]
    else:
        en_image = Path(tmpdir) / "english_test.png"
        _create_english_image(en_image)

    run_test("1. English Screenshot", en_image)

    # --- Test 2: Tamil screenshot ---
    ta_candidates = sorted(test_img_dir.glob("tamil_clean_*.png")) if test_img_dir.exists() else []
    if ta_candidates:
        ta_image = ta_candidates[0]
    else:
        ta_image = Path(tmpdir) / "tamil_test.png"
        _create_tamil_image(ta_image)

    run_test("2. Tamil Screenshot", ta_image)

    # --- Test 3: Tamil + English mixed screenshot ---
    mixed_candidates = sorted(test_img_dir.glob("mixed_tamil_en_*.png")) if test_img_dir.exists() else []
    if mixed_candidates:
        mixed_image = mixed_candidates[0]
    else:
        mixed_image = Path(tmpdir) / "mixed_test.png"
        _create_mixed_image(mixed_image)

    run_test("3. Tamil + English Mixed Screenshot", mixed_image)

    # --- Test 4: Very low quality / blank screenshot ---
    blank_image = Path(tmpdir) / "blank_test.png"
    _create_blank_image(blank_image)
    run_test("4. Very Low Quality / Blank Screenshot", blank_image)

    # --- Test 5: Invalid image (non-image binary file) ---
    invalid_image = Path(tmpdir) / "invalid.png"
    invalid_image.write_bytes(b"this is not an image")
    run_invalid_image_test("5. Invalid Image File", invalid_image)

    # --- Test 6: Missing file ---
    run_invalid_image_test("6. Missing File", Path(tmpdir) / "does_not_exist.png")

    print(f"\n{SEP}")
    print("OCR v2 Test Complete")
    print(SEP)

    # Cleanup
    import shutil
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
