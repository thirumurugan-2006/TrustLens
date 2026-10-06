import json
from pathlib import Path

import pytest
from PIL import Image

from app.input.loaders.screenshot_loader import ScreenshotParser


class FakeReader:
    def readtext(self, image_path: str, detail: int = 1):
        return [
            (None, "Congratulations!", 0.9),
            (None, "Pay registration fee.", 0.8),
        ]


def test_missing_image():
    with pytest.raises(FileNotFoundError):
        ScreenshotParser().parse("does_not_exist.png")


def test_invalid_image(tmp_path: Path):
    invalid_image = tmp_path / "not_an_image.png"
    invalid_image.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError):
        ScreenshotParser().parse(invalid_image)


def test_valid_image(tmp_path: Path):
    source_image = tmp_path / "test.png"
    Image.new("RGB", (200, 80), "white").save(source_image)

    parser = ScreenshotParser()
    parser._reader_instance = FakeReader()
    post = parser.parse(source_image)

    assert post.platform == "screenshot"
    assert post.text is not None
    assert post.images
    assert post.images[0].startswith("data/raw/screenshots/")
    assert post.metadata["ocr_engine"] == "easyocr_en+paddleocr_ta"
    assert post.metadata["ocr_confidence"] == pytest.approx(0.85)


def test_valid_image_saves_complete_json(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_image = tmp_path / "unicode-test.png"
    Image.new("RGB", (200, 80), "white").save(source_image)

    parser = ScreenshotParser()
    parser._reader_instance = FakeReader()
    post = parser.parse(source_image)

    json_path = Path("data/processed/normalized/screenshots") / f"{post.post_id}.json"
    assert json_path.exists()

    with json_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    assert json_path.stem == post.post_id
    assert data["post_id"] == post.post_id
    assert data["platform"] == "screenshot"
    assert "text" in data
    assert "images" in data
    assert "comments" in data
    assert "timestamp" in data
    assert "metadata" in data
    assert data["metadata"]["source_type"] == "screenshot"
    assert data["metadata"]["ocr_engine"] == "easyocr_en+paddleocr_ta"


def test_screenshot_metadata_includes_quality_and_image_info(tmp_path: Path):
    source_image = tmp_path / "metadata.png"
    Image.new("RGB", (300, 300), "white").save(source_image)

    class MediumReader:
        def readtext(self, image_path: str, detail: int = 1):
            return [(None, "usable text", 0.73)]

    parser = ScreenshotParser()
    parser._reader_instance = MediumReader()
    post = parser.parse(source_image)

    assert post.metadata["ocr_quality"] == "medium"
    assert post.metadata["ocr_warning"] is True
    assert post.metadata["image_format"] == "PNG"
    assert post.metadata["image_width"] == 300
    assert post.metadata["image_height"] == 300
    assert post.metadata["image_size_bytes"] > 0
