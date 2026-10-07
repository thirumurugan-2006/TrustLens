#!/usr/bin/env python3
"""
TrustLens T-6 Evidence Verification Training Script.

Trains a sequence pair classification model (XLM-R / MuRIL) to verify
claims against evidence documents under 4 relations:
SUPPORTS, CONTRADICTS, NEUTRAL, INSUFFICIENT.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


CLASSES = ["SUPPORTS", "CONTRADICTS", "NEUTRAL", "INSUFFICIENT"]
LABEL_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_LABEL = {i: c for i, c in enumerate(CLASSES)}


class ClaimEvidenceDataset(Dataset):
    """PyTorch Dataset for Claim-Evidence Verification."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def load_evidence_dataset(data_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Loads claim and evidence pairs from data/trustlens/ or data/processed/retrieval/.
    Extracts claim text, evidence text, languages, and 4 relation labels.
    """
    splits_data: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    lang_dist: Dict[str, int] = Counter()
    label_dist: Dict[str, int] = Counter()
    cross_lang_count = 0

    # 1. Check data/trustlens/ split structure
    is_trustlens_dir = (data_dir / "train" / "evidence.jsonl").exists() or (data_dir / "posts.jsonl").exists()

    if is_trustlens_dir:
        for sp in ["train", "validation", "test"]:
            claims_file = data_dir / sp / "claims.jsonl"
            ev_file = data_dir / sp / "evidence.jsonl"

            claims_map: Dict[str, Dict[str, Any]] = {}
            if claims_file.exists():
                with open(claims_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            c = json.loads(line)
                            claims_map[c["claim_id"]] = c

            if ev_file.exists():
                with open(ev_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        e = json.loads(line)
                        cid = e.get("claim_id")
                        c_data = claims_map.get(cid, {})

                        claim_text = c_data.get("claim_text", "")
                        ev_text = e.get("evidence_text", "")
                        rel = e.get("relation_label", "INSUFFICIENT")

                        if rel not in LABEL_TO_ID:
                            continue

                        c_lang = c_data.get("language", "unknown")
                        e_lang = e.get("language", "unknown")
                        is_cross = (c_lang != e_lang) and (c_lang != "unknown" and e_lang != "unknown")
                        if is_cross:
                            cross_lang_count += 1

                        rec = {
                            "pair_id": e.get("evidence_id"),
                            "claim_text": claim_text,
                            "evidence_text": ev_text,
                            "label_str": rel,
                            "label": LABEL_TO_ID[rel],
                            "claim_language": c_lang,
                            "evidence_language": e_lang,
                            "cross_language": is_cross,
                            "split": sp,
                        }
                        splits_data[sp].append(rec)
                        lang_dist[c_lang] += 1
                        label_dist[rel] += 1

    # 2. Check retrieval pairs fallback if trustlens split didn't yield records
    if sum(len(v) for v in splits_data.values()) == 0:
        retrieval_dir = data_dir
        for sp in ["train", "validation", "test"]:
            p_file = retrieval_dir / f"pairs_{sp}.jsonl"
            if p_file.exists():
                with open(p_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        p = json.loads(line)
                        rel = p.get("evidence_relation", "NEUTRAL")
                        if rel not in LABEL_TO_ID:
                            continue
                        is_cross = p.get("cross_language", False)
                        if is_cross:
                            cross_lang_count += 1
                        q_lang = p.get("query_language", "unknown")
                        rec = {
                            "pair_id": p.get("pair_id"),
                            "claim_text": p.get("query", ""),
                            "evidence_text": p.get("document_text", ""),
                            "label_str": rel,
                            "label": LABEL_TO_ID[rel],
                            "claim_language": q_lang,
                            "evidence_language": p.get("evidence_language", "unknown"),
                            "cross_language": is_cross,
                            "split": sp,
                        }
                        splits_data[sp].append(rec)
                        lang_dist[q_lang] += 1
                        label_dist[rel] += 1

    stats = {
        "total_pairs": sum(len(v) for v in splits_data.values()),
        "train_count": len(splits_data["train"]),
        "val_count": len(splits_data["validation"]),
        "test_count": len(splits_data["test"]),
        "cross_language_count": cross_lang_count,
        "languages": dict(lang_dist),
        "labels": dict(label_dist),
    }

    return splits_data, stats


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist()
    per_class = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def print_results(title: str, metrics: Dict[str, Any]):
    print("============================================================")
    print(title)
    print("============================================================")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Macro-F1:         {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:      {metrics['weighted_f1']:.4f}\n")
    print("Confusion Matrix:")
    header = "             " + "  ".join(f"{c[:4]:>7}" for c in CLASSES)
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        row_str = "  ".join(f"{v:>7}" for v in row)
        print(f"{CLASSES[i][:12]:<12} {row_str}")
    print("\nPer-class metrics:")
    for cls in CLASSES:
        cm_data = metrics["per_class"].get(cls, {})
        p = cm_data.get("precision", 0.0)
        r = cm_data.get("recall", 0.0)
        f = cm_data.get("f1-score", 0.0)
        sup = cm_data.get("support", 0)
        print(f"  {cls:<14} Precision: {p:.4f}  Recall: {r:.4f}  F1: {f:.4f}  Support: {sup}")
    print("============================================================\n")


def train_model(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # 1. Load Dataset
    splits_data, stats = load_evidence_dataset(data_dir)

    print("============================================================")
    print("TRUSTLENS T-6 EVIDENCE VERIFICATION")
    print("============================================================")
    print(f"Dataset:\n{data_dir.as_posix()}/\n")
    print(f"Train pairs:          {stats['train_count']}")
    print(f"Validation pairs:     {stats['val_count']}")
    print(f"Test pairs:           {stats['test_count']}")
    print(f"Cross-language pairs: {stats['cross_language_count']}\n")
    print("Labels:")
    for lbl in CLASSES:
        print(f"  {lbl:<14} {stats['labels'].get(lbl, 0)}")
    print("\nLanguage distribution:")
    for lng, cnt in stats["languages"].items():
        print(f"  {lng:<14} {cnt}")
    print(f"\nDevice:               {device}")
    print(f"Model backbone:       {args.model_name}")
    print("============================================================\n")

    # 2. Tokenizer & Statistics
    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    all_pairs = [r for sp in splits_data.values() for r in sp]
    token_lengths = [
        len(tokenizer.encode(r["claim_text"], r["evidence_text"], truncation=False))
        for r in all_pairs
    ]
    avg_tokens = sum(token_lengths) / max(len(token_lengths), 1)
    max_tokens = max(token_lengths) if token_lengths else 0
    truncated_count = sum(1 for l in token_lengths if l > args.max_length)
    truncation_rate = (truncated_count / max(len(token_lengths), 1)) * 100

    print("Tokenization statistics:")
    print(f"Average tokens:       {avg_tokens:.1f}")
    print(f"Maximum tokens:       {max_tokens}")
    print(f"Truncated:            {truncated_count}")
    print(f"Truncation rate:      {truncation_rate:.2f}%")
    print("============================================================\n")

    # 3. Model & DataLoader Setup
    torch.manual_seed(args.seed)
    print(f"Initializing {args.model_name} sequence classification model (num_labels=4)...")
    config = AutoConfig.from_pretrained(args.model_name, num_labels=len(CLASSES))
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config).to(device)

    def collate_fn(batch):
        claims = [b["claim_text"] for b in batch]
        evidence = [b["evidence_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        encodings = tokenizer(
            claims,
            evidence,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encodings["labels"] = labels
        encodings["pair_ids"] = [b.get("pair_id") for b in batch]
        return encodings

    train_loader = DataLoader(
        ClaimEvidenceDataset(splits_data["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ClaimEvidenceDataset(splits_data["validation"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        ClaimEvidenceDataset(splits_data["test"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 4. Training Loop
    best_val_f1 = -1.0
    best_val_metrics: Dict[str, Any] = {}
    saved_model_dir = output_dir / "models" / "evidence_verification"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    print("Beginning Training Loop:")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_steps = 0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            if step % args.logging_steps == 0 or step == len(train_loader):
                sys.stdout.write(f"\r  Epoch {epoch}/{args.epochs} Step {step}/{len(train_loader)} Current Batch Loss: {loss.item():.4f}")
                sys.stdout.flush()

        print()
        avg_train_loss = train_loss / max(train_steps, 1)

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_preds: List[int] = []
        val_targets: List[int] = []
        val_probs_list: List[List[float]] = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)

                val_probs_list.extend(probs.tolist())
                val_preds.extend(preds.tolist())
                val_targets.extend(labels.cpu().numpy().tolist())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_metrics = compute_metrics(val_targets, val_preds)

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Training Loss:        {avg_train_loss:.4f}")
        print(f"Validation Loss:      {avg_val_loss:.4f}")
        print(f"Validation Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"Validation Macro-F1:  {val_metrics['macro_f1']:.4f}")
        print("------------------------------------------------------------")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_val_metrics = val_metrics
            # Save best checkpoint
            model.save_pretrained(saved_model_dir)
            tokenizer.save_pretrained(saved_model_dir)

    print_results("EVIDENCE VERIFICATION VALIDATION RESULTS", best_val_metrics)

    # 5. Validation Probabilities Check
    val_probs_arr = np.array(val_probs_list)
    sums = np.sum(val_probs_arr, axis=1)
    sum_check = "PASS" if np.allclose(sums, 1.0, atol=1e-4) else "FAIL"
    print("============================================================")
    print("VALIDATION PROBABILITIES")
    print("============================================================")
    print(f"Rows:                   {len(val_probs_arr)}")
    print(f"Probability shape:      {val_probs_arr.shape}")
    print(f"Probability sum check:  {sum_check}")
    print("============================================================\n")

    # 6. Single Test Evaluation
    print("Evaluating Test set using best checkpoint...")
    model.eval()
    test_preds: List[int] = []
    test_targets: List[int] = []
    test_probs_list: List[List[float]] = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            test_probs_list.extend(probs.tolist())
            test_preds.extend(preds.tolist())
            test_targets.extend(batch["labels"].numpy().tolist())

    test_metrics = compute_metrics(test_targets, test_preds)
    print_results("EVIDENCE VERIFICATION TEST RESULTS", test_metrics)

    # 7. Save Predictions & Metrics
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    def save_predictions_jsonl(path: Path, records: List[Dict[str, Any]], probs: List[List[float]], preds: List[int]):
        with open(path, "w", encoding="utf-8") as f:
            for idx, r in enumerate(records):
                out = {
                    "pair_id": r.get("pair_id"),
                    "claim_language": r.get("claim_language"),
                    "evidence_language": r.get("evidence_language"),
                    "cross_language": r.get("cross_language"),
                    "true_label": r["label_str"],
                    "predicted_label": ID_TO_LABEL[preds[idx]],
                    "probabilities": {CLASSES[c]: float(probs[idx][c]) for c in range(len(CLASSES))},
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    val_pred_file = pred_dir / "evidence_verification_validation.jsonl"
    test_pred_file = pred_dir / "evidence_verification_test.jsonl"
    save_predictions_jsonl(val_pred_file, splits_data["validation"], val_probs_list, val_preds)
    save_predictions_jsonl(test_pred_file, splits_data["test"], test_probs_list, test_preds)

    metrics_file = saved_model_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({"validation": best_val_metrics, "test": test_metrics}, f, indent=2)

    print("============================================================")
    print("SAVED ARTIFACTS")
    print("============================================================")
    print(f"Model checkpoint:   {saved_model_dir}")
    print(f"Metrics:            {metrics_file}")
    print(f"Validation preds:   {val_pred_file}")
    print(f"Test preds:         {test_pred_file}")
    print("STATUS: COMPLETE")
    print("============================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustLens T-6 Evidence Verification")
    parser.add_argument("--data-dir", default="data/trustlens", help="Path to dataset directory")
    parser.add_argument("--output-dir", default="artifacts", help="Path to artifacts directory")
    parser.add_argument("--model-name", default="xlm-roberta-base", help="Pretrained model checkpoint (XLM-R / MuRIL)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length for (claim + evidence)")
    parser.add_argument("--logging-steps", type=int, default=20, help="Logging steps interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
