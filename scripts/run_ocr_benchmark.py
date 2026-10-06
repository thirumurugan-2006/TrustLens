from pathlib import Path
import argparse
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.benchmark.ocr_benchmark import BenchmarkSample, OCRResult, confidence_bucket, load_samples, sample_metrics, tamil_error_rows


class ExistingTrustLensOCR:
    name = "Existing OCR"
    language = "configured"

    def run(self, sample: BenchmarkSample) -> OCRResult:
        from app.input.loaders.screenshot_loader import ScreenshotParser
        started = time.perf_counter()
        try:
            post = ScreenshotParser().parse(sample.image_path)
            metadata = post.metadata
            return OCRResult(post.text or "", metadata.get("ocr_confidence"), (time.perf_counter() - started) * 1000, self.name, metadata.get("ocr_selected_language_mode", ["unknown"])[0], metadata.get("ocr_candidates", []), status="OK")
        except Exception as exc:
            return OCRResult(model_name=self.name, processing_time_ms=(time.perf_counter() - started) * 1000, error=str(exc), status="ERROR")


class ImprovedTamilOCR(ExistingTrustLensOCR):
    name = "Improved Tamil OCR"


class TesseractOCR:
    name = "Tesseract"
    language = "eng+tam"

    def __init__(self):
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.available = True
        except ImportError:
            self.pytesseract = None
            self.available = False

    def run(self, sample: BenchmarkSample) -> OCRResult:
        if not self.available:
            return OCRResult(model_name=self.name, language=self.language, status="NOT_IMPLEMENTED", error="pytesseract is not installed")
        started = time.perf_counter()
        try:
            text = self.pytesseract.image_to_string(str(sample.image_path), lang=self.language)
            return OCRResult(text=text, processing_time_ms=(time.perf_counter() - started) * 1000, model_name=self.name, language=self.language)
        except Exception as exc:
            return OCRResult(model_name=self.name, processing_time_ms=(time.perf_counter() - started) * 1000, error=str(exc), status="ERROR")


class TransformerOCR:
    name = "Transformer OCR"
    available = False

    def run(self, sample: BenchmarkSample) -> OCRResult:
        return OCRResult(model_name=self.name, status="NOT_IMPLEMENTED", error="No Transformer OCR implementation is installed")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], group_key: str | None = None) -> list[dict]:
    groups = {}
    for row in rows:
        if row["status"] != "OK" or row["annotation_status"] != "VALID":
            continue
        key = (row["model_name"], row[group_key]) if group_key else (row["model_name"],)
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in groups.items():
        model = key[0]
        group = key[1] if group_key else "overall"
        confidences = [row["confidence"] for row in values if row["confidence"] is not None]
        times = [row["processing_time_ms"] for row in values]
        output.append({
            "model_name": model,
            "category": group,
            "samples": len(values),
            "cer": sum(row["cer"] for row in values) / len(values),
            "wer": sum(row["wer"] for row in values) / len(values),
            "exact_match_accuracy": sum(row["exact_match"] for row in values) / len(values),
            "average_confidence": sum(confidences) / len(confidences) if confidences else None,
            "average_time_ms": sum(times) / len(times),
            "median_time_ms": sorted(times)[len(times) // 2],
            "images_per_second": 1000 / (sum(times) / len(times)) if times and sum(times) else None,
        })
    return output


def report_text(dataset: Path, samples: list[BenchmarkSample], missing: list[str], overall: list[dict], category: list[dict], statuses: list[dict]) -> str:
    lines = ["# OCR Benchmark Report", "", f"Dataset: `{dataset}`", f"Annotated samples: {sum(sample.ground_truth_text is not None for sample in samples)}", f"Samples discovered: {len(samples)}", ""]
    if missing:
        lines += ["## Missing Data", "", "Ground-truth annotations were missing or images were unavailable; those samples were excluded from supervised metrics.", ""] + [f"- {item}" for item in missing] + [""]
    lines += ["## Backend Status", "", "| Model | Status |", "|---|---|"] + [f"| {row['model_name']} | {row['status']} |" for row in statuses] + [""]
    lines += ["## Overall Metrics", "", "| Model | CER | WER | Exact Match | Avg ms |", "|---|---:|---:|---:|---:|"]
    for row in overall:
        lines.append(f"| {row['model_name']} | {row['cer']:.4f} | {row['wer']:.4f} | {row['exact_match_accuracy']:.4f} | {row['average_time_ms']:.2f} |")
    if not overall:
        lines.append("| No supervised results | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |")
    lines += ["", "## Category Metrics", "", "No category metrics are reported until annotated samples are available." if not category else "| Model | Category | CER | WER |\n|---|---|---:|---:|" ]
    for row in category:
        lines.append(f"| {row['model_name']} | {row['category']} | {row['cer']:.4f} | {row['wer']:.4f} |")
    lines += ["", "## Selection", "", "No model is recommended because no annotated benchmark samples produced supervised metrics." if not overall else "Recommendation is based on lowest CER, then WER, difficult-category performance, and inference time.", "", "## Limitations", "", "- Confidence calibration is descriptive only; no calibrated model was fitted.", "- Tesseract and Transformer OCR are reported as unavailable when their dependencies/implementations are absent.", "- No benchmark labels are used for training or OCR modification."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/ocr_test"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    output = args.output
    reports = Path("reports")
    output.mkdir(parents=True, exist_ok=True)
    (output / "error_samples").mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    if args.dataset.exists():
        samples, missing = load_samples(args.dataset)
    else:
        samples, missing = [], [f"Dataset directory missing: {args.dataset}"]
    engines = [ExistingTrustLensOCR(), TesseractOCR(), ImprovedTamilOCR(), TransformerOCR()]
    statuses = [{"model_name": engine.name, "status": "AVAILABLE" if getattr(engine, "available", True) else "NOT_IMPLEMENTED"} for engine in engines]
    rows, tamil_errors = [], []
    for sample in samples:
        for engine in engines:
            result = engine.run(sample)
            metrics = sample_metrics(result.text, sample.ground_truth_text)
            row = {"image_id": sample.image_id, "model_name": result.model_name, "predicted_text": result.text, "ground_truth": sample.ground_truth_text, "confidence": result.confidence, "processing_time_ms": result.processing_time_ms, "category": sample.category, "language": sample.language, "status": result.status, "error": result.error, **metrics, "confidence_bucket": confidence_bucket(result.confidence)}
            rows.append(row)
            if sample.ground_truth_text is not None:
                tamil_errors.extend(tamil_error_rows(sample.ground_truth_text, result.text, sample.image_id, result.model_name, sample.category))
    write_csv(output / "ocr_benchmark_results.csv", rows, list(rows[0].keys()) if rows else ["image_id", "model_name", "predicted_text", "ground_truth", "confidence", "processing_time_ms", "category", "language", "status", "error", "cer", "wer", "exact_match", "annotation_status", "confidence_bucket"])
    (output / "ocr_benchmark_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    overall = aggregate(rows)
    category = aggregate(rows, "category")
    write_csv(reports / "ocr_model_comparison.csv", overall, list(overall[0].keys()) if overall else ["model_name", "category", "samples", "cer", "wer", "exact_match_accuracy", "average_confidence", "average_time_ms", "median_time_ms", "images_per_second"])
    confidence_rows = []
    confidence_groups = {}
    for row in rows:
        if row["confidence"] is not None:
            key = (row["model_name"], row["confidence_bucket"])
            confidence_groups.setdefault(key, []).append(row)
    for (model_name, bucket), values in confidence_groups.items():
        confidence_rows.append({
            "model_name": model_name,
            "confidence_bucket": bucket,
            "samples": len(values),
            "average_cer": sum((value["cer"] or 0) for value in values) / len(values),
            "average_wer": sum((value["wer"] or 0) for value in values) / len(values),
        })
    write_csv(reports / "ocr_confidence_analysis.csv", confidence_rows, ["model_name", "confidence_bucket", "samples", "average_cer", "average_wer"])
    for row in rows:
        if row["status"] == "ERROR" or (row["cer"] is not None and row["cer"] > 0):
            error_path = output / "error_samples" / f"{row['model_name'].replace(' ', '_')}_{row['image_id']}.json"
            error_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(reports / "tamil_error_analysis.csv", tamil_errors, ["image_id", "model", "category", "ground_truth_char", "predicted_char", "count"])
    (reports / "ocr_benchmark_report.md").write_text(report_text(args.dataset, samples, missing, overall, category, statuses), encoding="utf-8")
    print(f"Samples: {len(samples)}")
    print(f"Missing annotations/data: {len(missing)}")
    print("No model recommendation without measured annotated results." if not overall else "Benchmark complete; see reports/ocr_benchmark_report.md")


if __name__ == "__main__":
    main()




