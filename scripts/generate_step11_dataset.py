"""
TrustLens Step 11 — Tamil OCR Test Dataset Generator
=====================================================

Generates a synthetic benchmark dataset of Tamil and mixed-script images
with ground truth annotations for Step 11 evaluation.

Categories covered:
  clean, blurry, noisy, low_resolution, mixed_tamil_english,
  stylized, multi_line, complex_background, numbers_symbols, social_media

Images are generated programmatically using PIL/Pillow. They are designed
to test real OCR difficulties, not to guarantee high accuracy.

Output: data/ocr_test/images/ + data/ocr_test/manifest.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    import numpy as np
except ImportError as exc:
    print(f"Missing dependency: {exc}\nInstall with: pip install pillow numpy")
    sys.exit(1)


OUTPUT_DIR = ROOT / "data" / "ocr_test"
IMAGE_DIR = OUTPUT_DIR / "images"
MANIFEST = OUTPUT_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Tamil text samples — real Tamil phrases used as ground truth
# These are NOT OCR outputs; they are verified Tamil Unicode strings.
# ---------------------------------------------------------------------------
TAMIL_SAMPLES = [
    "தமிழ்",
    "வணக்கம்",
    "இந்தியா",
    "செய்தி",
    "அரசு",
    "மக்கள்",
    "தமிழ்நாடு",
    "பள்ளி",
    "கல்வி",
    "நீதிமன்றம்",
]

MIXED_SAMPLES = [
    ("தமிழ் Nadu", "mixed"),
    ("India தமிழ்", "mixed"),
    ("COVID-19 தமிழ்நாடு", "mixed"),
    ("அரசு Order 2024", "mixed"),
]

NUMBER_SAMPLES = [
    "2024 ஆண்டு",
    "100 கோடி",
    "பக்கம் 42",
    "வரிசை 1",
]

MULTILINE_SAMPLES = [
    "வணக்கம்\nதமிழ்நாடு\nசெய்தி",
    "அரசு\nமக்கள்\nகல்வி",
]


def _get_font(size: int = 24):
    """Try to get a system font; fall back to PIL default."""
    font_candidates = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _render_text_image(
    text: str,
    size: tuple[int, int] = (400, 100),
    bg_color: tuple = (255, 255, 255),
    text_color: tuple = (0, 0, 0),
    font_size: int = 24,
    padding: int = 20,
) -> Image.Image:
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    draw.text((padding, padding), text, fill=text_color, font=font)
    return img


def _add_gaussian_noise(img: Image.Image, std: float = 25.0) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_dataset():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    idx = 0

    def save(img, image_id, ground_truth, language, category):
        nonlocal idx
        filename = f"{image_id}.png"
        img.save(IMAGE_DIR / filename)
        records.append({
            "image_id": image_id,
            "image": f"images/{filename}",
            "ground_truth_text": ground_truth,
            "language": language,
            "category": category,
            "optional_metadata": {},
        })
        idx += 1

    # ── Category 1: Clean Tamil ──────────────────────────────────────────────
    for i, text in enumerate(TAMIL_SAMPLES):
        img = _render_text_image(text, size=(400, 80), font_size=28)
        save(img, f"tamil_clean_{i+1:03d}", text, "ta", "clean")

    # ── Category 2: Blurry Tamil ─────────────────────────────────────────────
    for i, text in enumerate(TAMIL_SAMPLES[:5]):
        img = _render_text_image(text, size=(400, 80), font_size=28)
        img = img.filter(ImageFilter.GaussianBlur(radius=2.5))
        save(img, f"tamil_blurry_{i+1:03d}", text, "ta", "blurry")

    # ── Category 3: Noisy Tamil ──────────────────────────────────────────────
    for i, text in enumerate(TAMIL_SAMPLES[:5]):
        img = _render_text_image(text, size=(400, 80), font_size=28)
        img = _add_gaussian_noise(img, std=30.0)
        save(img, f"tamil_noisy_{i+1:03d}", text, "ta", "noisy")

    # ── Category 4: Low-resolution Tamil ─────────────────────────────────────
    for i, text in enumerate(TAMIL_SAMPLES[:5]):
        img = _render_text_image(text, size=(200, 50), font_size=14)
        # Downscale then upscale to simulate low-res screenshot
        small = img.resize((80, 20), Image.LANCZOS)
        img = small.resize((200, 50), Image.NEAREST)
        save(img, f"tamil_lowres_{i+1:03d}", text, "ta", "low_resolution")

    # ── Category 5: Mixed Tamil-English ─────────────────────────────────────
    for i, (text, lang) in enumerate(MIXED_SAMPLES):
        img = _render_text_image(text, size=(500, 80), font_size=24)
        save(img, f"mixed_tamil_en_{i+1:03d}", text, "mixed", "mixed_tamil_english")

    # ── Category 6: Numbers and symbols ──────────────────────────────────────
    for i, text in enumerate(NUMBER_SAMPLES):
        img = _render_text_image(text, size=(400, 80), font_size=24)
        save(img, f"tamil_numbers_{i+1:03d}", text, "ta", "numbers_symbols")

    # ── Category 7: Multi-line text ──────────────────────────────────────────
    for i, text in enumerate(MULTILINE_SAMPLES):
        img = _render_text_image(text, size=(400, 160), font_size=24)
        # For multiline ground truth, join with space for CER/WER
        gt = " ".join(text.split())
        save(img, f"tamil_multiline_{i+1:03d}", gt, "ta", "multi_line")

    # ── Category 8: Complex background ───────────────────────────────────────
    for i, text in enumerate(TAMIL_SAMPLES[:4]):
        # Simulate complex background with a gradient
        arr = np.zeros((80, 400, 3), dtype=np.uint8)
        for col in range(400):
            val = int(200 + 55 * (col / 400))
            arr[:, col, :] = [val, val - 30, val - 60]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        bg_img = Image.fromarray(arr)
        draw = ImageDraw.Draw(bg_img)
        font = _get_font(24)
        draw.text((20, 20), text, fill=(0, 0, 80), font=font)
        save(bg_img, f"tamil_complexbg_{i+1:03d}", text, "ta", "complex_background")

    # ── Category 9: Social-media screenshot style ─────────────────────────────
    for i, text in enumerate(TAMIL_SAMPLES[:4]):
        # Dark background, white text — typical social-media post
        img = _render_text_image(
            text,
            size=(600, 120),
            bg_color=(30, 30, 30),
            text_color=(240, 240, 240),
            font_size=30,
        )
        save(img, f"social_media_{i+1:03d}", text, "ta", "social_media")

    # ── Category 10: Stylized fonts (very bold) ───────────────────────────────
    for i, text in enumerate(TAMIL_SAMPLES[:4]):
        img = _render_text_image(text, size=(400, 100), font_size=36, bg_color=(250, 245, 235))
        save(img, f"tamil_stylized_{i+1:03d}", text, "ta", "stylized")

    # ── English baseline samples (for comparison) ─────────────────────────────
    english_samples = [
        ("Hello World", "clean"),
        ("Breaking News", "clean"),
        ("Tamil Nadu 2024", "mixed_tamil_english"),
        ("Test 123", "numbers_symbols"),
    ]
    for i, (text, category) in enumerate(english_samples):
        img = _render_text_image(text, size=(400, 80), font_size=28)
        save(img, f"english_baseline_{i+1:03d}", text, "en", category)

    MANIFEST.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dataset generated: {len(records)} samples in {OUTPUT_DIR}")
    print(f"Categories: {', '.join(sorted(set(r['category'] for r in records)))}")
    return records


if __name__ == "__main__":
    generate_dataset()
