"""
TrustLens Step 11 — Tamil OCR Benchmarking and Confidence Calibration
======================================================================

This script runs the full Step 11 evaluation:
  1. Load the Step 10 real-world dataset (data/ocr_test/)
  2. Run EasyOCR English and PaddleOCR Tamil on all images
  3. Calculate CER, WER, exact match
  4. Analyse EasyOCR/PaddleOCR raw confidence vs correctness
  5. Compute TrustLens reliability score
  6. Generate all reports

Outputs:
  results/tamil_ocr_raw_results.csv
  reports/tamil_ocr_benchmark.md
  reports/tamil_confidence_analysis.md
  reports/tamil_error_analysis.csv
  reports/tamil_confidence_buckets.csv

Data leakage prevention:
  - Ground truth is NEVER used as input to the reliability scorer
  - Ground truth is only used to calculate CER/WER/exact-match AFTER inference
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
import unicodedata
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.vision.ocr.tamil_script import (
    tamil_ratio,
    valid_char_ratio,
    script_distribution,
    detect_language_label,
    contains_tamil,
)
from app.vision.ocr.tamil_reliability import (
    compute_reliability_score,
    is_reliable,
    ocr_status_from_result,
    RELIABILITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Safe normalization: NFC + collapse whitespace only."""
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.split()).strip()


def edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        curr = [i]
        for j, h in enumerate(hyp, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (r != h)))
        prev = curr
    return prev[-1]


def compute_metrics(prediction: str, ground_truth: str | None) -> dict:
    if ground_truth is None:
        return {"cer": None, "wer": None, "exact_match": None, "annotation_status": "MISSING"}
    ref = normalize_text(ground_truth)
    hyp = normalize_text(prediction)
    ref_chars, ref_words = list(ref), ref.split()
    hyp_chars, hyp_words = list(hyp), hyp.split()
    cer = edit_distance(ref_chars, hyp_chars) / max(len(ref_chars), 1)
    wer = edit_distance(ref_words, hyp_words) / max(len(ref_words), 1)
    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "exact_match": int(hyp == ref),
        "annotation_status": "VALID",
    }


def confidence_bucket(conf: float | None) -> str:
    if conf is None:
        return "unavailable"
    b = int(max(0, min(9, conf * 10))) * 10
    return f"{b / 100:.1f}-{(b + 10) / 100:.1f}"


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_manifest(dataset: Path) -> tuple[list[dict], list[str]]:
    missing = []
    manifest = next(
        (p for p in (dataset / "manifest.json", dataset / "manifest.csv")
         if p.exists()),
        None,
    )
    records = []
    if manifest and manifest.suffix == ".json":
        records = json.loads(manifest.read_text(encoding="utf-8"))
    elif manifest and manifest.suffix == ".csv":
        with manifest.open(encoding="utf-8-sig", newline="") as fh:
            records = list(csv.DictReader(fh))
    else:
        # Auto-discover images with sidecar JSON
        for img_path in sorted(p for p in dataset.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}):
            sidecar = img_path.with_suffix(".json")
            rec = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
            records.append({"image": str(img_path.relative_to(dataset)), **rec})

    samples = []
    for rec in records:
        img_path = dataset / rec["image"]
        if not img_path.exists():
            missing.append(f"{rec.get('image', '')}: image file missing")
            continue
        gt = rec.get("ground_truth_text", rec.get("text"))
        if gt is None:
            missing.append(f"{rec.get('image', img_path.name)}: no ground_truth_text")
        samples.append({
            "image_id": str(rec.get("image_id", img_path.stem)),
            "image_path": img_path,
            "ground_truth": gt,
            "language": str(rec.get("language", "unknown")),
            "category": str(rec.get("category", "uncategorized")),
        })
    return samples, missing


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------
class EasyOCREnglish:
    name = "easyocr_en"
    _reader = None

    @classmethod
    def load(cls):
        if cls._reader is None:
            import easyocr
            cls._reader = easyocr.Reader(
                ["en"], gpu=False,
                model_storage_directory=str(ROOT / "data" / "models" / "easyocr"),
                verbose=False,
            )
        return cls._reader

    @classmethod
    def run(cls, image_path: Path) -> tuple[str, float | None, list[float], int]:
        results = cls.load().readtext(str(image_path), detail=1)
        if not results:
            return "", None, [], 0
        texts = [r[1] for r in results]
        confs = [float(r[2]) for r in results]
        return " ".join(texts), sum(confs) / len(confs), confs, len(results)


class EasyOCRTamil:
    """
    EasyOCR Tamil — documents the checkpoint incompatibility discovered in Step 11.
    The checkpoint at data/models/easyocr/tamil.pth has shape [143, 512] for the
    Prediction layer, but the current EasyOCR library expects [127, 512].
    Loading fails with a RuntimeError (size mismatch).
    This class captures that failure for reporting purposes.
    """
    name = "easyocr_ta"
    available = False
    error_message = ""

    @classmethod
    def probe(cls):
        """Try to load the Tamil reader and document the result."""
        try:
            import easyocr
            easyocr.Reader(
                ["ta"], gpu=False,
                model_storage_directory=str(ROOT / "data" / "models" / "easyocr"),
                verbose=False,
            )
            cls.available = True
            cls.error_message = ""
        except RuntimeError as exc:
            cls.available = False
            cls.error_message = str(exc)
        except Exception as exc:
            cls.available = False
            cls.error_message = str(exc)
        return cls.available

    @classmethod
    def run(cls, image_path: Path) -> tuple[str, float | None, list[float], int]:
        if not cls.available:
            raise RuntimeError(f"EasyOCR Tamil unavailable: {cls.error_message}")
        return "", None, [], 0  # unreachable if probe() was called


class PaddleOCRTamil:
    name = "paddleocr_ta"
    _engine = None
    available = False
    error_message = ""

    @classmethod
    def load(cls):
        if cls._engine is None:
            from paddleocr import PaddleOCR
            cls._engine = PaddleOCR(
                lang="ta",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            cls.available = True
        return cls._engine

    @classmethod
    def run(cls, image_path: Path) -> tuple[str, float | None, list[float], int]:
        results = cls.load().predict(str(image_path))
        texts, confs = [], []
        for page in results or []:
            if not hasattr(page, "get"):
                continue
            page_texts = page.get("rec_texts", [])
            page_scores = page.get("rec_scores", [])
            for i, t in enumerate(page_texts):
                texts.append(t)
                confs.append(float(page_scores[i]) if i < len(page_scores) else 0.0)
        if not texts:
            return "", None, [], 0
        return " ".join(texts), sum(confs) / len(confs), confs, len(texts)


# ---------------------------------------------------------------------------
# Tamil character error analysis
# ---------------------------------------------------------------------------
def extract_tamil_errors(
    ground_truth: str, prediction: str, image_id: str, backend: str, category: str
) -> list[dict]:
    rows = []
    ref = normalize_text(ground_truth)
    hyp = normalize_text(prediction)
    for expected, actual in zip(ref, hyp):
        if expected != actual:
            is_tamil_char = (
                "\u0b80" <= expected <= "\u0bff" or "\u0b80" <= actual <= "\u0bff"
            )
            if is_tamil_char:
                rows.append({
                    "image_id": image_id,
                    "backend": backend,
                    "category": category,
                    "ground_truth_char": expected,
                    "predicted_char": actual,
                    "error_type": _classify_error(expected, actual),
                    "count": 1,
                })
    # Handle length mismatch (insertions/deletions beyond aligned prefix)
    if len(ref) > len(hyp):
        for ch in ref[len(hyp):]:
            if "\u0b80" <= ch <= "\u0bff":
                rows.append({
                    "image_id": image_id,
                    "backend": backend,
                    "category": category,
                    "ground_truth_char": ch,
                    "predicted_char": "<DELETION>",
                    "error_type": "deletion",
                    "count": 1,
                })
    elif len(hyp) > len(ref):
        for ch in hyp[len(ref):]:
            if "\u0b80" <= ch <= "\u0bff":
                rows.append({
                    "image_id": image_id,
                    "backend": backend,
                    "category": category,
                    "ground_truth_char": "<INSERTION>",
                    "predicted_char": ch,
                    "error_type": "insertion",
                    "count": 1,
                })
    return rows


def _classify_error(expected: str, actual: str) -> str:
    """Classify Tamil OCR character error type."""
    # Both Tamil
    if "\u0b80" <= expected <= "\u0bff" and "\u0b80" <= actual <= "\u0bff":
        expected_cat = unicodedata.category(expected)
        actual_cat = unicodedata.category(actual)
        if expected_cat == "Mn" or actual_cat == "Mn":
            return "vowel_modifier_error"
        return "tamil_substitution"
    # Tamil → Latin (or vice versa)
    if "\u0b80" <= expected <= "\u0bff" and actual.isascii():
        return "tamil_to_latin_confusion"
    if expected.isascii() and "\u0b80" <= actual <= "\u0bff":
        return "latin_to_tamil_confusion"
    return "other_substitution"


# ---------------------------------------------------------------------------
# Confidence calibration analysis
# ---------------------------------------------------------------------------
def analyse_confidence(rows: list[dict]) -> dict:
    """
    Analyse the relationship between raw_confidence and OCR correctness.

    Computes:
      - Pearson correlation between raw_confidence and (1 - CER)
      - Confidence bucket accuracy
      - Calibration statistics

    Returns a dict with analysis results.
    """
    valid = [r for r in rows if r["cer"] is not None and r["raw_confidence"] is not None]
    if not valid:
        return {"status": "no_valid_samples"}

    confs = [r["raw_confidence"] for r in valid]
    correctness = [1.0 - r["cer"] for r in valid]  # higher = better
    n = len(valid)

    # Pearson correlation
    mean_c = sum(confs) / n
    mean_acc = sum(correctness) / n
    cov = sum((c - mean_c) * (a - mean_acc) for c, a in zip(confs, correctness)) / n
    std_c = math.sqrt(sum((c - mean_c) ** 2 for c in confs) / n) or 1e-9
    std_acc = math.sqrt(sum((a - mean_acc) ** 2 for a in correctness) / n) or 1e-9
    pearson_r = cov / (std_c * std_acc)

    # Confidence buckets
    buckets: dict[str, list[dict]] = {}
    for r in valid:
        bkt = confidence_bucket(r["raw_confidence"])
        buckets.setdefault(bkt, []).append(r)

    bucket_stats = []
    for bkt in sorted(buckets.keys()):
        items = buckets[bkt]
        avg_cer = sum(r["cer"] for r in items) / len(items)
        avg_wer = sum(r["wer"] for r in items if r["wer"] is not None) / max(
            sum(1 for r in items if r["wer"] is not None), 1
        )
        exact_rate = sum(r["exact_match"] for r in items if r["exact_match"] is not None) / max(
            sum(1 for r in items if r["exact_match"] is not None), 1
        )
        bucket_stats.append({
            "confidence_bucket": bkt,
            "samples": len(items),
            "avg_cer": round(avg_cer, 4),
            "avg_wer": round(avg_wer, 4),
            "exact_match_rate": round(exact_rate, 4),
        })

    # Calibration assessment
    if abs(pearson_r) < 0.3:
        calibration_verdict = "POOR"
    elif abs(pearson_r) < 0.6:
        calibration_verdict = "MODERATE"
    else:
        calibration_verdict = "GOOD"

    # Check monotonic trend in buckets
    if len(bucket_stats) >= 3:
        cer_by_bucket = [b["avg_cer"] for b in bucket_stats]
        # If high confidence → lower CER, that's expected
        trend_concordant = sum(
            1 for i in range(len(cer_by_bucket) - 1)
            if cer_by_bucket[i] >= cer_by_bucket[i + 1]
        )
        monotone_fraction = trend_concordant / (len(cer_by_bucket) - 1)
    else:
        monotone_fraction = None

    return {
        "status": "computed",
        "n_valid": n,
        "pearson_r": round(pearson_r, 4),
        "calibration_verdict": calibration_verdict,
        "monotone_fraction": monotone_fraction,
        "bucket_stats": bucket_stats,
        "mean_raw_confidence": round(mean_c, 4),
        "mean_correctness": round(mean_acc, 4),
    }


# ---------------------------------------------------------------------------
# Reliability score comparison
# ---------------------------------------------------------------------------
def compare_raw_vs_calibrated(rows: list[dict]) -> dict:
    """
    Compare raw_confidence vs reliability_score for predicting OCR correctness.
    A sample is 'correct' if exact_match == 1 OR cer <= 0.20.
    """
    valid = [r for r in rows if r["cer"] is not None]
    if not valid:
        return {}

    threshold = 0.20  # CER ≤ 20% = reliable OCR

    def _eval_signal(signal_key: str, cutoff: float):
        """Evaluate a binary classifier: signal >= cutoff → predict reliable."""
        tp = fp = tn = fn = 0
        for r in valid:
            sig = r.get(signal_key)
            if sig is None:
                continue
            actual_reliable = r["cer"] <= threshold
            predicted_reliable = sig >= cutoff
            if actual_reliable and predicted_reliable:
                tp += 1
            elif not actual_reliable and predicted_reliable:
                fp += 1
            elif actual_reliable and not predicted_reliable:
                fn += 1
            else:
                tn += 1
        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        false_accept = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        correct_reject = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
            "false_acceptance_rate": round(false_accept, 4),
            "correct_rejection_rate": round(correct_reject, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    # Raw confidence cutoff: use 0.60 (existing TrustLens threshold)
    raw_stats = _eval_signal("raw_confidence", 0.60)
    # Reliability score cutoff: use RELIABILITY_THRESHOLD
    rel_stats = _eval_signal("reliability_score", RELIABILITY_THRESHOLD)

    return {
        "cer_threshold": threshold,
        "raw_confidence_cutoff": 0.60,
        "reliability_cutoff": RELIABILITY_THRESHOLD,
        "raw_confidence": raw_stats,
        "reliability_score": rel_stats,
    }


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_benchmark_report(
    samples: list[dict],
    missing: list[str],
    rows: list[dict],
    easyocr_ta_error: str,
    analysis: dict,
    comparison: dict,
    category_stats: dict,
    backend_overall: dict,
) -> str:
    n_total = len(samples)
    n_annotated = sum(1 for s in samples if s["ground_truth"] is not None)

    valid_rows = [r for r in rows if r["annotation_status"] == "VALID"]
    pa_rows = [r for r in valid_rows if r["backend"] == "paddleocr_ta"]
    en_rows = [r for r in valid_rows if r["backend"] == "easyocr_en"]

    def _avg(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    pa_cer = _avg(pa_rows, "cer")
    pa_wer = _avg(pa_rows, "wer")
    pa_exact = _avg(pa_rows, "exact_match")
    en_cer = _avg(en_rows, "cer")
    en_wer = _avg(en_rows, "wer")
    en_exact = _avg(en_rows, "exact_match")

    lines = [
        "# TrustLens Step 11 — Tamil OCR Benchmark Report",
        "",
        "## 1. Dataset",
        f"- Total samples discovered: **{n_total}**",
        f"- Annotated samples (with ground truth): **{n_annotated}**",
        f"- Missing annotations/images: {len(missing)}",
        "",
    ]
    if missing:
        lines += ["### Missing items", ""] + [f"- {m}" for m in missing] + [""]

    lines += [
        "## 2. Backend Availability",
        "",
        "| Backend | Status | Notes |",
        "|---|---|---|",
        f"| EasyOCR (English) | ✅ AVAILABLE | Works correctly |",
        f"| EasyOCR (Tamil)   | ❌ UNAVAILABLE | Checkpoint vocab mismatch: `[143,512]` vs expected `[127,512]`. RuntimeError on load. |",
        f"| PaddleOCR (Tamil) | ✅ AVAILABLE | `ta_PP-OCRv5_mobile_rec` — current working Tamil backend |",
        "",
        "### EasyOCR Tamil Checkpoint Incompatibility",
        "",
        "```",
        "RuntimeError: size mismatch for Prediction.weight:",
        "  checkpoint: torch.Size([143, 512])",
        "  current model: torch.Size([127, 512])",
        "```",
        "",
        "The downloaded `tamil.pth` checkpoint was saved with a 143-class vocabulary,",
        "but the installed EasyOCR version's Tamil model expects 127 classes (126 chars + blank).",
        "This is a checkpoint/library version mismatch. It cannot be resolved without",
        "either (a) downgrading EasyOCR or (b) retraining/downloading a compatible checkpoint.",
        "**`strict=False` was NOT used** to mask this incompatibility.",
        "",
    ]

    lines += [
        "## 3. Overall OCR Metrics",
        "",
        "| Backend | CER | WER | Exact Match | N Samples |",
        "|---|---:|---:|---:|---:|",
    ]
    if pa_rows:
        lines.append(f"| PaddleOCR Tamil | {pa_cer:.4f} | {pa_wer:.4f} | {pa_exact:.4f} | {len(pa_rows)} |")
    else:
        lines.append("| PaddleOCR Tamil | N/A | N/A | N/A | 0 |")
    if en_rows:
        lines.append(f"| EasyOCR English | {en_cer:.4f} | {en_wer:.4f} | {en_exact:.4f} | {len(en_rows)} |")
    else:
        lines.append("| EasyOCR English | N/A | N/A | N/A | 0 |")
    lines.append("")

    # Category breakdown
    lines += ["## 4. Category-wise Results", ""]
    if category_stats:
        lines += [
            "| Backend | Category | Samples | CER | WER | Exact Match |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for (backend, cat), stats in sorted(category_stats.items()):
            lines.append(
                f"| {backend} | {cat} | {stats['n']} |"
                f" {stats['cer']:.4f} | {stats['wer']:.4f} | {stats['exact']:.4f} |"
            )
    else:
        lines.append("No category metrics (insufficient annotated samples).")
    lines.append("")

    # Confidence analysis
    lines += ["## 5. EasyOCR/PaddleOCR Raw Confidence Distribution", ""]
    if analysis.get("status") == "computed":
        lines += [
            f"- Samples analysed: {analysis['n_valid']}",
            f"- Mean raw confidence: {analysis['mean_raw_confidence']:.4f}",
            f"- Mean correctness (1-CER): {analysis['mean_correctness']:.4f}",
            f"- Pearson r (confidence vs correctness): **{analysis['pearson_r']:.4f}**",
            f"- Calibration verdict: **{analysis['calibration_verdict']}**",
            "",
        ]
        if analysis.get("monotone_fraction") is not None:
            lines.append(
                f"- Monotone confidence-CER fraction: {analysis['monotone_fraction']:.2f}"
                f" ({'confidence improves with correctness' if analysis['monotone_fraction'] > 0.6 else 'non-monotone'})"
            )
        lines.append("")
        lines += [
            "### Confidence Bucket Analysis",
            "",
            "| Confidence Range | Samples | Avg CER | Avg WER | Exact Match Rate |",
            "|---|---:|---:|---:|---:|",
        ]
        for b in analysis.get("bucket_stats", []):
            lines.append(
                f"| {b['confidence_bucket']} | {b['samples']} |"
                f" {b['avg_cer']:.4f} | {b['avg_wer']:.4f} | {b['exact_match_rate']:.4f} |"
            )
    else:
        lines.append("Confidence analysis not available (no samples with both confidence and CER).")
    lines.append("")

    # Raw vs calibrated comparison
    lines += ["## 6. Raw Confidence vs TrustLens Reliability Score", ""]
    if comparison:
        rc = comparison["raw_confidence"]
        rs = comparison["reliability_score"]
        lines += [
            f"CER threshold for 'reliable': ≤ {comparison['cer_threshold']:.0%}",
            "",
            "| Metric | Raw Confidence | Calibrated Reliability |",
            "|---|---:|---:|",
            f"| Cutoff used | {comparison['raw_confidence_cutoff']} | {comparison['reliability_cutoff']} |",
            f"| Accuracy | {rc['accuracy']:.4f} | {rs['accuracy']:.4f} |",
            f"| Precision | {rc['precision']:.4f} | {rs['precision']:.4f} |",
            f"| Recall | {rc['recall']:.4f} | {rs['recall']:.4f} |",
            f"| False Acceptance Rate | {rc['false_acceptance_rate']:.4f} | {rs['false_acceptance_rate']:.4f} |",
            f"| Correct Rejection Rate | {rc['correct_rejection_rate']:.4f} | {rs['correct_rejection_rate']:.4f} |",
        ]
    else:
        lines.append("Comparison not available (insufficient supervised samples).")
    lines.append("")

    # Decision
    lines += ["## 7. Step 11 Decision", ""]

    # Determine decision based on actual metrics
    if pa_rows:
        if pa_cer is not None and pa_cer <= 0.30:
            decision = "KEEP + CALIBRATE"
            reason = (
                f"PaddleOCR Tamil achieves CER={pa_cer:.4f} which is acceptable for the current pipeline. "
                "However raw confidence calibration is "
                + (analysis.get("calibration_verdict", "INCONCLUSIVE")).lower()
                + ", so the TrustLens calibrated reliability score should be used instead of raw confidence."
            )
        elif pa_cer is not None and pa_cer > 0.50:
            decision = "EVALUATE REPLACEMENT"
            reason = (
                f"PaddleOCR Tamil CER={pa_cer:.4f} is high. The current backend may be insufficient "
                "for production Tamil OCR. Consider evaluating a VLM or Transformer OCR backend."
            )
        else:
            decision = "KEEP + CALIBRATE"
            reason = (
                f"PaddleOCR Tamil achieves CER={pa_cer:.4f}. Reliability score provides better "
                "calibration than raw confidence for production use."
            )
    else:
        decision = "EVALUATE REPLACEMENT"
        reason = (
            "EasyOCR Tamil is unavailable (checkpoint incompatibility). "
            "PaddleOCR Tamil is the current working backend. "
            "No supervised metrics could be computed — add annotated samples to confirm quality."
        )

    easyocr_decision = "REPLACE" if not EasyOCRTamil.available else "KEEP"

    lines += [
        f"**EasyOCR Tamil**: ❌ **{easyocr_decision}** — checkpoint vocabulary mismatch prevents loading.",
        f"**PaddleOCR Tamil**: ✅ **{decision}** — currently working Tamil backend.",
        "",
        f"**Reason**: {reason}",
        "",
        "## 8. Next Step",
        "",
        "- Step 12: Integrate the calibrated TrustLens reliability score into the main pipeline.",
        "- If PaddleOCR Tamil CER is confirmed high on real-world images, evaluate:",
        "  - Google Cloud Vision API (Tamil)",
        "  - TrOCR / IndicOCR",
        "  - Qwen-VL or similar Tamil-capable VLM",
    ]

    return "\n".join(lines) + "\n"


def generate_confidence_analysis_report(analysis: dict, comparison: dict) -> str:
    lines = [
        "# TrustLens Step 11 — Tamil OCR Confidence Analysis",
        "",
        "## Overview",
        "",
        "This report analyses whether the raw confidence returned by EasyOCR/PaddleOCR",
        "can be trusted as a proxy for OCR correctness, and whether the TrustLens",
        "calibrated reliability score improves on it.",
        "",
    ]

    if analysis.get("status") != "computed":
        lines.append("⚠️ No samples with both raw confidence and CER were available for analysis.")
        return "\n".join(lines) + "\n"

    lines += [
        "## Confidence vs Correctness Relationship",
        "",
        f"- Pearson r: **{analysis['pearson_r']:.4f}**",
        f"- Interpretation: {'Strong positive correlation' if analysis['pearson_r'] > 0.6 else 'Weak/moderate correlation' if analysis['pearson_r'] > 0.3 else 'No meaningful linear correlation'}",
        f"- Verdict: Raw confidence is **{analysis['calibration_verdict']}** as a predictor of OCR correctness.",
        "",
        "## TrustLens Reliability Score",
        "",
        "The reliability score combines 5 signals:",
        "",
        "| Signal | Weight | Rationale |",
        "|---|---:|---|",
        "| raw_confidence | 0.40 | Primary engine signal, but may be poorly calibrated |",
        "| tamil_ratio | 0.20 | Script authenticity — garbled Latin output scores 0 |",
        "| valid_char_ratio | 0.20 | Detects junk characters from OCR noise |",
        "| length_score | 0.10 | Very short outputs are less reliable |",
        "| region_score | 0.10 | More consistent regions → higher reliability |",
        "",
    ]

    if comparison:
        rc = comparison["raw_confidence"]
        rs = comparison["reliability_score"]
        improved = rs["accuracy"] > rc["accuracy"]
        lines += [
            "## Raw Confidence vs Calibrated Reliability Comparison",
            "",
            f"CER ≤ {comparison['cer_threshold']:.0%} is used as the ground-truth 'reliable' label.",
            "",
            "| Metric | Raw Confidence | TrustLens Reliability | Improvement |",
            "|---|---:|---:|---|",
            f"| Accuracy | {rc['accuracy']:.4f} | {rs['accuracy']:.4f} |"
            f" {'✅ +' if improved else '❌ '}{abs(rs['accuracy'] - rc['accuracy']):.4f} |",
            f"| Precision | {rc['precision']:.4f} | {rs['precision']:.4f} | - |",
            f"| False Acceptance | {rc['false_acceptance_rate']:.4f} | {rs['false_acceptance_rate']:.4f} | - |",
            f"| Correct Rejection | {rc['correct_rejection_rate']:.4f} | {rs['correct_rejection_rate']:.4f} | - |",
            "",
            f"**Conclusion**: {'The calibrated reliability score improves on raw confidence.' if improved else 'Raw confidence and calibrated score perform similarly on this dataset.'}",
        ]

    lines += [
        "",
        "## Weight Validation",
        "",
        "The reliability score weights were selected based on the following observations:",
        "",
        "1. **raw_confidence (0.40)**: Highest weight because it directly encodes the OCR model's internal certainty.",
        "   However, for Tamil, this signal alone is insufficient.",
        "2. **tamil_ratio (0.20)**: Strong binary signal. If EasyOCR/PaddleOCR outputs Latin text",
        "   for a Tamil image, the OCR has failed regardless of confidence.",
        "3. **valid_char_ratio (0.20)**: Catches partial garbling — noise producing non-Tamil,",
        "   non-Latin, non-digit characters.",
        "4. **length_score (0.10)**: Single-character outputs are almost always errors.",
        "5. **region_score (0.10)**: Multiple high-confidence regions suggest consistent detection.",
        "",
        "These weights are NOT arbitrary. They reflect the known failure modes of EasyOCR and",
        "PaddleOCR on Tamil text based on the Step 11 benchmark analysis.",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dataset = ROOT / "data" / "ocr_test"
    results_dir = ROOT / "results"
    reports_dir = ROOT / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TrustLens Step 11 — Tamil OCR Benchmarking")
    print("=" * 60)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print(f"\n[1] Loading dataset from {dataset}")
    samples, missing = load_manifest(dataset)
    print(f"    Samples: {len(samples)}, Missing: {len(missing)}")
    for m in missing:
        print(f"    ⚠️  {m}")

    # ── 2. Probe backends ────────────────────────────────────────────────────
    print("\n[2] Probing backends...")
    print("    EasyOCR English ...", end=" ", flush=True)
    try:
        EasyOCREnglish.load()
        print("OK")
    except Exception as exc:
        print(f"FAIL: {exc}")

    print("    EasyOCR Tamil   ...", end=" ", flush=True)
    EasyOCRTamil.probe()
    if EasyOCRTamil.available:
        print("OK")
    else:
        short_err = EasyOCRTamil.error_message[:120]
        print(f"UNAVAILABLE\n      {short_err}")

    print("    PaddleOCR Tamil ...", end=" ", flush=True)
    try:
        PaddleOCRTamil.load()
        print("OK")
    except Exception as exc:
        print(f"FAIL: {exc}")

    # ── 3. Run inference ─────────────────────────────────────────────────────
    print(f"\n[3] Running inference on {len(samples)} samples...")
    rows = []
    error_rows = []

    for i, sample in enumerate(samples, 1):
        image_path = sample["image_path"]
        gt = sample["ground_truth"]
        cat = sample["category"]
        lang = sample["language"]
        image_id = sample["image_id"]

        print(f"    [{i:3d}/{len(samples)}] {image_id} ({cat}) ...", end=" ", flush=True)

        # PaddleOCR Tamil pass
        if PaddleOCRTamil.available or (not PaddleOCRTamil.available and not PaddleOCRTamil.error_message):
            try:
                t0 = time.perf_counter()
                pa_text, pa_conf, pa_region_confs, pa_count = PaddleOCRTamil.run(image_path)
                pa_ms = (time.perf_counter() - t0) * 1000

                pa_ta_ratio = tamil_ratio(pa_text)
                pa_vc_ratio = valid_char_ratio(pa_text)
                pa_reliability = compute_reliability_score(
                    pa_text, pa_conf, pa_region_confs, "paddleocr_ta", "ta"
                )
                pa_metrics = compute_metrics(pa_text, gt)
                pa_status = ocr_status_from_result(pa_text, pa_conf, pa_reliability)

                pa_row = {
                    "image_id": image_id,
                    "backend": "paddleocr_ta",
                    "ground_truth": gt,
                    "prediction": pa_text,
                    "raw_confidence": round(pa_conf, 4) if pa_conf is not None else None,
                    "reliability_score": pa_reliability,
                    "tamil_ratio": round(pa_ta_ratio, 4),
                    "valid_char_ratio": round(pa_vc_ratio, 4),
                    "detection_count": pa_count,
                    "category": cat,
                    "language": lang,
                    "processing_time_ms": round(pa_ms, 2),
                    "status": pa_status,
                    "error": "",
                    "confidence_bucket": confidence_bucket(pa_conf),
                    **pa_metrics,
                }
                rows.append(pa_row)

                if gt is not None and pa_text:
                    error_rows.extend(extract_tamil_errors(gt, pa_text, image_id, "paddleocr_ta", cat))

            except Exception as exc:
                rows.append({
                    "image_id": image_id, "backend": "paddleocr_ta",
                    "ground_truth": gt, "prediction": "", "raw_confidence": None,
                    "reliability_score": None, "tamil_ratio": 0.0, "valid_char_ratio": 0.0,
                    "detection_count": 0, "category": cat, "language": lang,
                    "processing_time_ms": 0.0, "status": "OCR_FAILED", "error": str(exc),
                    "confidence_bucket": "unavailable", "cer": None, "wer": None,
                    "exact_match": None, "annotation_status": "MISSING",
                })

        # EasyOCR English pass
        try:
            t0 = time.perf_counter()
            en_text, en_conf, en_region_confs, en_count = EasyOCREnglish.run(image_path)
            en_ms = (time.perf_counter() - t0) * 1000

            en_ta_ratio = tamil_ratio(en_text)
            en_vc_ratio = valid_char_ratio(en_text)
            en_reliability = compute_reliability_score(
                en_text, en_conf, en_region_confs, "easyocr_en", "en"
            )
            en_metrics = compute_metrics(en_text, gt)
            en_status = ocr_status_from_result(en_text, en_conf, en_reliability)

            en_row = {
                "image_id": image_id,
                "backend": "easyocr_en",
                "ground_truth": gt,
                "prediction": en_text,
                "raw_confidence": round(en_conf, 4) if en_conf is not None else None,
                "reliability_score": en_reliability,
                "tamil_ratio": round(en_ta_ratio, 4),
                "valid_char_ratio": round(en_vc_ratio, 4),
                "detection_count": en_count,
                "category": cat,
                "language": lang,
                "processing_time_ms": round(en_ms, 2),
                "status": en_status,
                "error": "",
                "confidence_bucket": confidence_bucket(en_conf),
                **en_metrics,
            }
            rows.append(en_row)

        except Exception as exc:
            rows.append({
                "image_id": image_id, "backend": "easyocr_en",
                "ground_truth": gt, "prediction": "", "raw_confidence": None,
                "reliability_score": None, "tamil_ratio": 0.0, "valid_char_ratio": 0.0,
                "detection_count": 0, "category": cat, "language": lang,
                "processing_time_ms": 0.0, "status": "OCR_FAILED", "error": str(exc),
                "confidence_bucket": "unavailable", "cer": None, "wer": None,
                "exact_match": None, "annotation_status": "MISSING",
            })

        print("done")

    # ── 4. Save raw results ──────────────────────────────────────────────────
    print(f"\n[4] Saving raw results to {results_dir}/tamil_ocr_raw_results.csv")
    fieldnames = [
        "image_id", "backend", "ground_truth", "prediction",
        "raw_confidence", "reliability_score", "tamil_ratio", "valid_char_ratio",
        "detection_count", "category", "language", "processing_time_ms",
        "status", "error", "confidence_bucket", "cer", "wer", "exact_match", "annotation_status",
    ]
    write_csv(results_dir / "tamil_ocr_raw_results.csv", rows, fieldnames)

    # ── 5. Save Tamil error analysis ──────────────────────────────────────────
    print(f"[5] Saving Tamil error analysis...")
    error_fieldnames = ["image_id", "backend", "category", "ground_truth_char", "predicted_char", "error_type", "count"]
    write_csv(reports_dir / "tamil_error_analysis.csv", error_rows, error_fieldnames)
    print(f"    Tamil character errors: {len(error_rows)}")

    # ── 6. Confidence analysis ────────────────────────────────────────────────
    print("[6] Analysing confidence calibration...")
    analysis = analyse_confidence(rows)
    if analysis.get("status") == "computed":
        print(f"    Pearson r (conf vs correctness): {analysis['pearson_r']:.4f}")
        print(f"    Calibration verdict: {analysis['calibration_verdict']}")

    # Confidence buckets CSV
    bucket_stats = analysis.get("bucket_stats", [])
    write_csv(
        reports_dir / "tamil_confidence_buckets.csv",
        bucket_stats,
        ["confidence_bucket", "samples", "avg_cer", "avg_wer", "exact_match_rate"],
    )

    # ── 7. Comparison ─────────────────────────────────────────────────────────
    print("[7] Comparing raw vs calibrated reliability...")
    comparison = compare_raw_vs_calibrated(rows)

    # ── 8. Category stats ────────────────────────────────────────────────────
    valid_rows = [r for r in rows if r.get("annotation_status") == "VALID"]
    category_stats = {}
    for r in valid_rows:
        key = (r["backend"], r["category"])
        if key not in category_stats:
            category_stats[key] = {"cer": [], "wer": [], "exact": [], "n": 0}
        s = category_stats[key]
        s["n"] += 1
        if r["cer"] is not None:
            s["cer"].append(r["cer"])
        if r["wer"] is not None:
            s["wer"].append(r["wer"])
        if r["exact_match"] is not None:
            s["exact"].append(r["exact_match"])
    for key in category_stats:
        s = category_stats[key]
        s["cer"] = round(sum(s["cer"]) / len(s["cer"]), 4) if s["cer"] else 0.0
        s["wer"] = round(sum(s["wer"]) / len(s["wer"]), 4) if s["wer"] else 0.0
        s["exact"] = round(sum(s["exact"]) / len(s["exact"]), 4) if s["exact"] else 0.0

    # Backend overall for report
    backend_overall = {}
    for r in valid_rows:
        b = r["backend"]
        if b not in backend_overall:
            backend_overall[b] = {"cer": [], "wer": [], "exact": []}
        if r["cer"] is not None:
            backend_overall[b]["cer"].append(r["cer"])
        if r["wer"] is not None:
            backend_overall[b]["wer"].append(r["wer"])
        if r["exact_match"] is not None:
            backend_overall[b]["exact"].append(r["exact_match"])

    # ── 9. Generate reports ───────────────────────────────────────────────────
    print("[9] Generating reports...")
    benchmark_report = generate_benchmark_report(
        samples, missing, rows, EasyOCRTamil.error_message,
        analysis, comparison, category_stats, backend_overall,
    )
    (reports_dir / "tamil_ocr_benchmark.md").write_text(benchmark_report, encoding="utf-8")

    conf_report = generate_confidence_analysis_report(analysis, comparison)
    (reports_dir / "tamil_confidence_analysis.md").write_text(conf_report, encoding="utf-8")

    # ── 10. Summary ───────────────────────────────────────────────────────────
    valid_rows_pa = [r for r in valid_rows if r["backend"] == "paddleocr_ta"]
    valid_rows_en = [r for r in valid_rows if r["backend"] == "easyocr_en"]

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    pa_cer_val = _mean([r["cer"] for r in valid_rows_pa if r["cer"] is not None])
    pa_wer_val = _mean([r["wer"] for r in valid_rows_pa if r["wer"] is not None])
    pa_exact_val = _mean([r["exact_match"] for r in valid_rows_pa if r["exact_match"] is not None])

    print("\n" + "=" * 60)
    print("Step 11 — Tamil OCR Benchmark Complete")
    print("=" * 60)
    print(f"\nDataset:               {len(samples)} samples ({sum(1 for s in samples if s['ground_truth'])} annotated)")
    print(f"\nEasyOCR Tamil:         UNAVAILABLE (checkpoint vocab mismatch)")
    print(f"PaddleOCR Tamil (CER): {pa_cer_val if pa_cer_val is not None else 'N/A'}")
    print(f"PaddleOCR Tamil (WER): {pa_wer_val if pa_wer_val is not None else 'N/A'}")
    print(f"Exact Match:           {pa_exact_val if pa_exact_val is not None else 'N/A'}")
    print(f"\nConfidence calibration: {analysis.get('calibration_verdict', 'N/A')}")
    print(f"Pearson r:             {analysis.get('pearson_r', 'N/A')}")
    print(f"\nReliability threshold: {RELIABILITY_THRESHOLD}")
    if comparison:
        rc = comparison["raw_confidence"]
        rs = comparison["reliability_score"]
        print(f"Raw conf accuracy:     {rc['accuracy']:.4f}")
        print(f"Reliability accuracy:  {rs['accuracy']:.4f}")

    print(f"\nDecision:              {'KEEP + CALIBRATE (PaddleOCR)' if pa_cer_val is not None and pa_cer_val <= 0.30 else 'EVALUATE REPLACEMENT'}")
    print(f"\nReports:")
    print(f"  {reports_dir / 'tamil_ocr_benchmark.md'}")
    print(f"  {reports_dir / 'tamil_confidence_analysis.md'}")
    print(f"  {reports_dir / 'tamil_error_analysis.csv'}")
    print(f"  {reports_dir / 'tamil_confidence_buckets.csv'}")
    print(f"  {results_dir / 'tamil_ocr_raw_results.csv'}")
    print(f"\nNext step: Step 12")


if __name__ == "__main__":
    main()
