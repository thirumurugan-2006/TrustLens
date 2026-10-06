from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes import input as routes_input
from app.main import app


class FakeReader:
    def readtext(self, image_path: str, detail: int = 1):
        return [(None, "Screenshot text", 0.95)]


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_text_endpoint():
    response = client.post(
        "/api/input/text",
        json={"text": "Earn ₹50,000 per month. Pay ₹500 registration fee."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["platform"] == "text"
    assert data["text"]
    assert data["metadata"]


def test_empty_text_returns_400():
    response = client.post("/api/input/text", json={"text": "   "})

    assert response.status_code == 400


def test_screenshot_endpoint(tmp_path, monkeypatch):
    image_path = tmp_path / "upload.png"
    Image.new("RGB", (120, 60), "white").save(image_path)
    routes_input.pipeline.router.screenshot._reader_instance = FakeReader()

    with image_path.open("rb") as image_file:
        response = client.post(
            "/api/input/screenshot",
            files={"file": ("upload.png", image_file, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["platform"] == "screenshot"
    assert data["text"]
    assert data["images"]
    assert data["metadata"]


def test_invalid_screenshot_returns_400():
    response = client.post(
        "/api/input/screenshot",
        files={"file": ("not-image.png", BytesIO(b"not an image"), "image/png")},
    )

    assert response.status_code == 400


def test_reddit_endpoint_handles_unavailable_api():
    response = client.post(
        "/api/input/reddit",
        json={
            "url": "https://www.reddit.com/r/example/comments/abc123/example/"
        },
    )

    assert response.status_code in (200, 503)


def test_invalid_reddit_url_returns_400():
    response = client.post("/api/input/reddit", json={"url": "https://google.com"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid Reddit post URL."


def test_small_screenshot_returns_warning(tmp_path):
    image_path = tmp_path / "small.png"
    Image.new("RGB", (50, 50), "white").save(image_path)
    routes_input.pipeline.router.screenshot._reader_instance = FakeReader()

    with image_path.open("rb") as image_file:
        response = client.post(
            "/api/input/screenshot",
            files={"file": ("small.png", image_file, "image/png")},
        )

    assert response.status_code == 200


def test_text_endpoint_includes_language_analysis():
    response = client.post(
        "/api/input/text",
        json={"text": "This is a normal English sentence."},
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["language_analysis"]["primary_language"] == "en"


def test_screenshot_endpoint_includes_language_analysis(tmp_path):
    image_path = tmp_path / "language.png"
    Image.new("RGB", (120, 60), "white").save(image_path)
    routes_input.pipeline.router.screenshot._reader_instance = FakeReader()

    with image_path.open("rb") as image_file:
        response = client.post(
            "/api/input/screenshot",
            files={"file": ("language.png", image_file, "image/png")},
        )

    assert response.status_code == 200
    assert "language_analysis" in response.json()["metadata"]


def test_screenshot_endpoint_exposes_multilingual_ocr_metadata(tmp_path):
    image_path = tmp_path / "metadata.png"
    Image.new("RGB", (300, 300), "white").save(image_path)
    routes_input.pipeline.router.screenshot._reader_instance = FakeReader()

    with image_path.open("rb") as image_file:
        response = client.post(
            "/api/input/screenshot",
            files={"file": ("metadata.png", image_file, "image/png")},
        )

    metadata = response.json()["metadata"]
    assert response.status_code == 200
    assert metadata["ocr_languages"] == ["en", "ta"]
    assert "ocr_confidence" in metadata
    assert "ocr_quality" in metadata
    assert "ocr_unreliable" in metadata
    assert "language_analysis" in metadata
