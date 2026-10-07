#!/usr/bin/env python3
"""
TrustLens T-1 Claim Detection Training Script — XLM-RoBERTa.

Fine-tunes xlm-roberta-base for binary claim detection (CLAIM vs. NON_CLAIM)
across multilingual social media posts. Prints full terminal metrics and saves artifacts.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


CLASSES = ["NON_CLAIM", "CLAIM"]
LABEL_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_LABEL = {i: c for i, c in enumerate(CLASSES)}


class PostClaimDataset(Dataset):
    """PyTorch Dataset for Claim Detection."""

    def __init__(self, texts: List[str], labels: List[int], post_ids: List[str], languages: List[str]):
        self.texts = texts
        self.labels = labels
        self.post_ids = post_ids
        self.languages = languages

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
            "post_id": self.post_ids[idx],
            "language": self.languages[idx],
        }


def load_claim_dataset(data_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Loads posts from data/trustlens/posts.jsonl and annotations from annotated/claims.jsonl.
    Preserves existing train/validation/test splits.
    """
    posts_file = data_dir / "posts.jsonl"
    claims_file = data_dir / "annotated" / "claims.jsonl"

    if not posts_file.exists():
        raise FileNotFoundError(f"posts.jsonl not found in {data_dir}")

    # Load post annotations for claims
    claims_map: Dict[str, str] = {}
    if claims_file.exists():
        with open(claims_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    pid = item.get("post_id")
                    lbl = item.get("detection_label", "CLAIM")
                    if pid:
                        claims_map[pid] = "CLAIM" if lbl == "CLAIM" else "NON_CLAIM"

    splits_data: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    lang_dist: Dict[str, int] = Counter()
    label_dist: Dict[str, int] = Counter()

    with open(posts_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            pid = p.get("post_id")
            sp = p.get("split_info", {}).get("split", "train")
            if sp not in splits_data:
                sp = "train"

            # Extract post content text
            content = p.get("content", {})
            text = content.get("text") or p.get("text") or ""
            if content.get("title"):
                text = f"{content['title']} {text}".strip()

            lang = p.get("language_info", {}).get("primary", "unknown")
            label_str = claims_map.get(pid, "CLAIM")
            label_id = LABEL_TO_ID.get(label_str, 1)

            record = {
                "post_id": pid,
                "text": text,
                "label": label_id,
                "label_str": label_str,
                "language": lang,
                "split": sp,
            }
            splits_data[sp].append(record)
            lang_dist[lang] += 1
            label_dist[label_str] += 1

    stats = {
        "total_posts": sum(len(v) for v in splits_data.values()),
        "train_count": len(splits_data["train"]),
        "val_count": len(splits_data["validation"]),
        "test_count": len(splits_data["test"]),
        "languages": dict(lang_dist),
        "labels": dict(label_dist),
    }

    return splits_data, stats


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    per_class = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
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
    header = "       " + "  ".join(f"{c:>10}" for c in CLASSES)
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"{CLASSES[i]:<7} {row_str}")
    print("============================================================\n")


def train_model(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # 1. Load Data
    splits_data, stats = load_claim_dataset(data_dir)

    print("============================================================")
    print("TRUSTLENS T-1 XLM-R CLAIM DETECTION")
    print("============================================================")
    print(f"Dataset:\n{data_dir.as_posix()}/\n")
    print(f"Train:            {stats['train_count']}")
    print(f"Validation:       {stats['val_count']}")
    print(f"Test:             {stats['test_count']}\n")
    print("Labels:")
    for lbl, cnt in stats["labels"].items():
        print(f"  {lbl:<12} {cnt}")
    print("\nLanguage distribution:")
    for lng, cnt in stats["languages"].items():
        print(f"  {lng:<12} {cnt}")
    print(f"\nDevice:           {device}")
    print(f"Model:            {args.model_name}")
    print("============================================================\n")

    # 2. Tokenizer & Statistics
    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    all_texts = [r["text"] for sp in splits_data.values() for r in sp]
    token_lengths = [len(tokenizer.encode(t, truncation=False)) for t in all_texts]
    avg_tokens = sum(token_lengths) / max(len(token_lengths), 1)
    max_tokens = max(token_lengths) if token_lengths else 0
    truncated_count = sum(1 for l in token_lengths if l > args.max_length)
    truncation_rate = (truncated_count / max(len(token_lengths), 1)) * 100

    print("Tokenization statistics:")
    print(f"Average tokens:   {avg_tokens:.1f}")
    print(f"Maximum tokens:   {max_tokens}")
    print(f"Truncated:        {truncated_count}")
    print(f"Truncation rate:  {truncation_rate:.2f}%")
    print("============================================================\n")

    # 3. Model & DataLoader Setup
    torch.manual_seed(args.seed)
    print(f"Initializing {args.model_name} sequence classification model...")
    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config).to(device)

    def collate_fn(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encodings["labels"] = labels
        encodings["post_ids"] = [b["post_id"] for b in batch]
        return encodings

    train_ds = PostClaimDataset(
        [r["text"] for r in splits_data["train"]],
        [r["label"] for r in splits_data["train"]],
        [r["post_id"] for r in splits_data["train"]],
        [r["language"] for r in splits_data["train"]],
    )
    val_ds = PostClaimDataset(
        [r["text"] for r in splits_data["validation"]],
        [r["label"] for r in splits_data["validation"]],
        [r["post_id"] for r in splits_data["validation"]],
        [r["language"] for r in splits_data["validation"]],
    )
    test_ds = PostClaimDataset(
        [r["text"] for r in splits_data["test"]],
        [r["label"] for r in splits_data["test"]],
        [r["post_id"] for r in splits_data["test"]],
        [r["language"] for r in splits_data["test"]],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 4. Training Loop
    best_val_f1 = -1.0
    best_val_metrics: Dict[str, Any] = {}
    saved_model_dir = output_dir / "models" / "claim_detection" / "xlm_roberta"
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

    print_results("XLM-R VALIDATION RESULTS", best_val_metrics)

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
    print_results("XLM-R TEST RESULTS", test_metrics)

    # 7. Save Predictions & Metrics
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    def save_predictions_jsonl(path: Path, records: List[Dict[str, Any]], probs: List[List[float]], preds: List[int]):
        with open(path, "w", encoding="utf-8") as f:
            for idx, r in enumerate(records):
                out = {
                    "post_id": r["post_id"],
                    "language": r["language"],
                    "true_label": r["label_str"],
                    "predicted_label": ID_TO_LABEL[preds[idx]],
                    "probabilities": {CLASSES[c]: float(probs[idx][c]) for c in range(len(CLASSES))},
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    val_pred_file = pred_dir / "claim_detection_xlm_roberta_validation.jsonl"
    test_pred_file = pred_dir / "claim_detection_xlm_roberta_test.jsonl"
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
    parser = argparse.ArgumentParser(description="TrustLens T-1 XLM-RoBERTa Claim Detection")
    parser.add_argument("--data-dir", default="data/trustlens", help="Path to trustlens data directory")
    parser.add_argument("--output-dir", default="artifacts", help="Path to artifacts directory")
    parser.add_argument("--model-name", default="xlm-roberta-base", help="Pretrained model checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--logging-steps", type=int, default=20, help="Logging steps interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
