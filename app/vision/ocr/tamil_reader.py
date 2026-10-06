from pathlib import Path


class TamilOCR:
    backend_name = "paddleocr"

    def __init__(self, model_directory: str | Path = "data/models/paddleocr") -> None:
        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        self._engine = None
        self.error = None

    def _load(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                self.error = "PaddleOCR/PaddlePaddle is not installed"
                raise RuntimeError(self.error) from exc
            try:
                self._engine = PaddleOCR(lang="ta")
            except Exception as exc:
                self.error = str(exc)
                raise RuntimeError(f"PaddleOCR Tamil backend unavailable: {exc}") from exc
        return self._engine

    def readtext(self, image_path: str, detail: int = 1):
        result = self._load().predict(image_path)
        detections = []
        for page in result or []:
            texts = page.get("rec_texts", []) if hasattr(page, "get") else []
            scores = page.get("rec_scores", []) if hasattr(page, "get") else []
            boxes = page.get("rec_boxes", []) if hasattr(page, "get") else []
            for index, text in enumerate(texts):
                confidence = float(scores[index]) if index < len(scores) else 0.0
                box = boxes[index] if index < len(boxes) else None
                detections.append((box, text, confidence))
        return detections
