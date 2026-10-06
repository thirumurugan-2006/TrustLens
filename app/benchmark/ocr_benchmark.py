from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import re
import statistics
import time
import unicodedata


@dataclass
class OCRResult:
    text: str = ""
    confidence: float | None = None
    processing_time_ms: float = 0.0
    model_name: str = ""
    language: str = "unknown"
    regions: list = None
    error: str = ""
    status: str = "OK"

    def __post_init__(self):
        if self.regions is None:
            self.regions = []

    def to_dict(self):
        return asdict(self)


@dataclass
class BenchmarkSample:
    image_id: str
    image_path: Path
    ground_truth_text: str | None
    language: str
    category: str
    optional_metadata: dict


def normalize_for_comparison(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.split()).strip()


def edit_distance(reference: list, hypothesis: list) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, 1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (ref_item != hyp_item)))
        previous = current
    return previous[-1]


def sample_metrics(prediction: str, ground_truth: str | None) -> dict:
    if ground_truth is None:
        return {"cer": None, "wer": None, "exact_match": None, "annotation_status": "MISSING"}
    reference = normalize_for_comparison(ground_truth)
    predicted = normalize_for_comparison(prediction)
    reference_chars = list(reference)
    reference_words = reference.split()
    return {
        "cer": edit_distance(reference_chars, list(predicted)) / max(len(reference_chars), 1),
        "wer": edit_distance(reference_words, predicted.split()) / max(len(reference_words), 1),
        "exact_match": int(predicted == reference),
        "annotation_status": "VALID",
    }


def load_samples(dataset: Path) -> tuple[list[BenchmarkSample], list[str]]:
    manifest = next((path for path in (dataset / "manifest.json", dataset / "manifest.csv", dataset / "annotations.jsonl") if path.exists()), None)
    records = []
    missing = []
    if manifest and manifest.suffix == ".json":
        records = json.loads(manifest.read_text(encoding="utf-8"))
    elif manifest and manifest.suffix == ".jsonl":
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif manifest and manifest.suffix == ".csv":
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
    else:
        for image in sorted(path for path in dataset.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}):
            sidecar = image.with_suffix(".json")
            record = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
            records.append({"image": str(image.relative_to(dataset)), **record})

    samples = []
    for index, record in enumerate(records):
        image_path = dataset / record["image"]
        if not image_path.exists():
            missing.append(f"{record.get('image', '')}: image missing")
            continue
        ground_truth = record.get("ground_truth_text", record.get("text"))
        if ground_truth is None:
            missing.append(f"{record.get('image', image_path.name)}: ground_truth_text missing")
        samples.append(BenchmarkSample(
            image_id=str(record.get("image_id", image_path.stem or index)),
            image_path=image_path,
            ground_truth_text=ground_truth,
            language=str(record.get("language", "unknown")),
            category=str(record.get("category", "uncategorized")),
            optional_metadata=record.get("optional_metadata", {}),
        ))
    return samples, missing


def tamil_error_rows(reference: str, prediction: str, image_id: str, model: str, category: str) -> list[dict]:
    rows = []
    for expected, actual in zip(reference, prediction):
        if expected != actual and ("\u0b80" <= expected <= "\u0bff" or "\u0b80" <= actual <= "\u0bff"):
            rows.append({"image_id": image_id, "model": model, "category": category, "ground_truth_char": expected, "predicted_char": actual, "count": 1})
    return rows


def confidence_bucket(confidence: float | None) -> str:
    if confidence is None:
        return "unavailable"
    lower = int(max(0, min(9, confidence * 10))) * 10
    return f"{lower / 100:.2f}-{(lower + 10) / 100:.2f}"
