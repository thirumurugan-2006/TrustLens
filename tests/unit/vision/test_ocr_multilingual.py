from pathlib import Path

from PIL import Image

from app.input.loaders.screenshot_loader import ScreenshotParser
from app.preprocessing.language.language_analyzer import LanguageAnalyzer


class SequenceReader:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def readtext(self, image_path: str, detail: int = 1):
        self.calls += 1
        return self.results[min(self.calls - 1, len(self.results) - 1)]


def make_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "ocr.png"
    Image.new("RGB", (300, 300), "white").save(image_path)
    return image_path


def parse_with_results(tmp_path: Path, results):
    parser = ScreenshotParser()
    parser._reader_instance = SequenceReader(results)
    return parser.parse(make_image(tmp_path))


def test_english_ocr_regression(tmp_path):
    post = parse_with_results(
        tmp_path,
        [[(None, "Earn 500 every month", 0.88)]],
    )

    assert post.text == "Earn 500 every month"
    assert post.metadata["ocr_languages"] == ["en", "ta"]
    assert post.metadata["ocr_quality"] == "high"
    assert post.metadata["ocr_unreliable"] is False
    assert post.metadata["ocr_attempts"] == 1
    assert post.metadata["language_analysis"]["primary_language"] == "en"


def test_tamil_ocr_detects_tamil_script(tmp_path):
    post = parse_with_results(
        tmp_path,
        [[(None, "உங்கள் முதல் சம்பளம்", 0.86)]],
    )

    analysis = post.metadata["language_analysis"]
    assert "Tamil" in analysis["scripts"]
    assert analysis["primary_language"] == "ta"
    assert analysis["language_analysis_reliable"] is True


def test_mixed_tamil_english_ocr(tmp_path):
    post = parse_with_results(
        tmp_path,
        [[(None, "இந்த job மிகவும் நல்லது", 0.84)]],
    )

    analysis = post.metadata["language_analysis"]
    assert "Tamil" in analysis["scripts"]
    assert "Latin" in analysis["scripts"]
    assert analysis["is_code_mixed"] is True


def test_low_confidence_ocr_is_not_called_english(tmp_path):
    class MockService:
        def english_available(self): return True
        def tamil_available(self): return True
        def run_ocr(self, img, mode, prep):
            return {"text": "20g61 Bou BDVomd", "confidence": 0.25, "language_mode": mode, "preprocessing": prep, "detection_count": 1}

    parser = ScreenshotParser()
    parser._get_pipeline().service = MockService()
    post = parser.parse(make_image(tmp_path))

    metadata = post.metadata
    assert metadata["ocr_unreliable"] is True
    assert metadata["language_analysis"]["primary_language"] == "unknown"
    assert metadata["language_analysis"]["language_analysis_reliable"] is False


def test_empty_ocr_is_unreliable(tmp_path):
    post = parse_with_results(tmp_path, [[], []])

    assert post.text == ""
    assert post.metadata["ocr_confidence"] == 0.0
    assert post.metadata["ocr_unreliable"] is True
    assert post.metadata["language_analysis"]["primary_language"] == "unknown"


def test_tamil_and_tanglish_direct_text():
    analyzer = LanguageAnalyzer()
    tamil = analyzer.analyze("உங்கள் முதல் சம்பளத்தை எப்படி முதலீடு செய்வது?")
    tanglish = analyzer.analyze("Indha investment romba nalla irukku, monthly SIP start pannunga.")

    assert tamil["primary_language"] == "ta"
    assert tamil["scripts"] == ["Tamil"]
    assert tamil["language_analysis_reliable"] is True
    assert tanglish["scripts"] == ["Latin"]
    assert tanglish["transliteration_candidate"] is True
    assert tanglish["transliteration_language"] == "ta"


def test_unusable_english_cannot_beat_reliable_tamil(tmp_path):
    class MockService:
        def english_available(self): return True
        def tamil_available(self): return True
        def run_ocr(self, img, mode, prep):
            if mode == ["en"]:
                res = {"text": "20g61 Bou BDVomd", "confidence": 0.265, "language_mode": ["en"], "preprocessing": prep, "detection_count": 1}
            else:
                res = {"text": "தமிழ் முதலீடு", "confidence": 0.72, "language_mode": ["ta"], "preprocessing": prep, "detection_count": 1}
            return res

    parser = ScreenshotParser()
    parser._get_pipeline().service = MockService()
    post = parser.parse(make_image(tmp_path))

    metadata = post.metadata
    assert metadata["ocr_selected_language_mode"] == ["ta"]
    assert metadata["ocr_unreliable"] is False
