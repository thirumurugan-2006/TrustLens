#!/usr/bin/env python3
"""
TrustLens T-8 LightGBM Risk Classifier Training Script.

Trains, evaluates, calibrates, and exports the 4-class tabular risk classifier
(HIGH, MEDIUM, LOW, INSUFFICIENT) using the 92-feature structured matrix.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


TARGET_CLASSES = ["HIGH", "MEDIUM", "LOW", "INSUFFICIENT"]
LABEL_TO_INT = {c: i for i, c in enumerate(TARGET_CLASSES)}
INT_TO_LABEL = {i: c for i, c in enumerate(TARGET_CLASSES)}


def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], str]:
    """Loads feature datasets and validates schema and target column."""
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"
    schema_path = data_dir / "feature_schema.json"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Feature dataset CSV files not found in {data_dir}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    target_col = "risk_label"
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df_train.columns.tolist()}")

    # Determine 92 feature columns
    metadata_cols = {"feature_id", "post_id", "cluster_id", "split", target_col}
    feature_cols = [c for c in df_train.columns if c not in metadata_cols]

    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
            expected_features = [f["name"] for f in schema_data.get("features", [])]
            if expected_features:
                feature_cols = [c for c in expected_features if c in df_train.columns]

    if len(feature_cols) != 92:
        raise ValueError(f"Expected exactly 92 feature columns, found {len(feature_cols)}")

    if target_col in feature_cols:
        raise ValueError(f"Target column '{target_col}' must not be present in feature columns!")

    return df_train, df_val, df_test, feature_cols, target_col


def print_dataset_info(
    data_dir: Path,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
):
    """Prints comprehensive pre-training dataset information."""
    train_counts = Counter(df_train[target_col])
    val_counts = Counter(df_val[target_col])
    test_counts = Counter(df_test[target_col])

    missing_train = df_train[feature_cols].isna().sum().sum()
    missing_val = df_val[feature_cols].isna().sum().sum()
    missing_test = df_test[feature_cols].isna().sum().sum()

    print("============================================================")
    print("TRUSTLENS T-8 LIGHTGBM TRAINING")
    print("============================================================")
    print(f"Dataset:\n{data_dir.as_posix()}/\n")
    print(f"Train rows:       {len(df_train)}")
    print(f"Validation rows:  {len(df_val)}")
    print(f"Test rows:        {len(df_test)}")
    print(f"\nFeature count:    {len(feature_cols)}")
    print(f"Target column:    {target_col}\n")
    print("Classes:")
    for cls in TARGET_CLASSES:
        print(f"  {cls:<12} Train: {train_counts.get(cls, 0):<5} Val: {val_counts.get(cls, 0):<4} Test: {test_counts.get(cls, 0):<4}")
    print(f"\nMissing values:   Train={missing_train}, Val={missing_val}, Test={missing_test}")
    print("============================================================\n")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes multiclass evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_CLASSES)))).tolist()
    per_class = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_CLASSES))),
        target_names=TARGET_CLASSES,
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


def print_evaluation_results(title: str, metrics: Dict[str, Any]):
    """Pretty-prints evaluation results."""
    print("============================================================")
    print(title)
    print("============================================================")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Macro-F1:         {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:      {metrics['weighted_f1']:.4f}\n")
    print("Confusion Matrix:")
    header = "       " + "  ".join(f"{c[:4]:>5}" for c in TARGET_CLASSES)
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        row_str = "  ".join(f"{v:>5}" for v in row)
        print(f"{TARGET_CLASSES[i][:6]:<6} {row_str}")
    print("\nPer-class metrics:")
    for cls in TARGET_CLASSES:
        cm_data = metrics["per_class"].get(cls, {})
        p = cm_data.get("precision", 0.0)
        r = cm_data.get("recall", 0.0)
        f = cm_data.get("f1-score", 0.0)
        sup = cm_data.get("support", 0)
        print(f"  {cls:<12} Precision: {p:.4f}  Recall: {r:.4f}  F1: {f:.4f}  Support: {sup}")
    print("============================================================\n")


def train_and_evaluate(args: argparse.Namespace):
    """Executes LightGBM training, validation, testing, and artifact exports."""
    # Check LightGBM installation
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError:
        print("\n============================================================")
        print("LIGHTGBM PACKAGE MISSING")
        print("Please install it using:")
        print("pip install lightgbm")
        print("============================================================\n")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    df_train, df_val, df_test, feature_cols, target_col = load_dataset(data_dir)
    print_dataset_info(data_dir, df_train, df_val, df_test, feature_cols, target_col)

    # Encode targets
    y_train = np.array([LABEL_TO_INT[v] for v in df_train[target_col]])
    y_val = np.array([LABEL_TO_INT[v] for v in df_val[target_col]])
    y_test = np.array([LABEL_TO_INT[v] for v in df_test[target_col]])

    # Encode any categorical columns for LightGBM
    X_train = df_train[feature_cols].copy()
    X_val = df_val[feature_cols].copy()
    X_test = df_test[feature_cols].copy()

    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = pd.Categorical(X_val[col], categories=X_train[col].cat.categories)
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

    # LightGBM Classifier
    print("Training LightGBM model...")
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(TARGET_CLASSES),
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        random_state=args.seed,
        class_weight="balanced",
        verbose=1,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "validation"],
        callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True)],
    )

    # Validation evaluation
    val_probs = clf.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    val_metrics = compute_metrics(y_val, val_preds)
    print_evaluation_results("VALIDATION RESULTS", val_metrics)

    # Validation Probability Verification
    prob_sums = np.sum(val_probs, axis=1)
    sum_check = "PASS" if np.allclose(prob_sums, 1.0, atol=1e-5) else "FAIL"
    print("============================================================")
    print("VALIDATION PROBABILITIES")
    print("============================================================")
    print(f"Rows:                   {len(val_probs)}")
    print(f"Probability shape:      {val_probs.shape}")
    print(f"Probability sum check:  {sum_check}")
    print("============================================================\n")

    # Test evaluation
    test_probs = clf.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_metrics = compute_metrics(y_test, test_preds)
    print_evaluation_results("TEST RESULTS", test_metrics)

    # Save artifacts
    model_dir = output_dir / "models" / "risk" / "lightgbm"
    pred_dir = output_dir / "predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    clf.booster_.save_model((model_dir / "model.txt").as_posix())

    # Save config
    config = {
        "model_type": "LightGBM",
        "task": "T-8 Risk Classifier",
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "target_classes": TARGET_CLASSES,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "seed": args.seed,
            "best_iteration": int(clf.best_iteration_) if clf.best_iteration_ else args.n_estimators,
        },
    }
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Save metrics
    all_metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "best_iteration": int(clf.best_iteration_) if clf.best_iteration_ else args.n_estimators,
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Save predictions JSONL
    def save_predictions_jsonl(path: Path, df_orig: pd.DataFrame, probs: np.ndarray, preds: np.ndarray):
        with open(path, "w", encoding="utf-8") as f:
            for idx, (_, row) in enumerate(df_orig.iterrows()):
                record = {
                    "post_id": row.get("post_id"),
                    "true_label": row.get(target_col),
                    "predicted_label": INT_TO_LABEL[preds[idx]],
                    "probabilities": {TARGET_CLASSES[c]: float(probs[idx][c]) for c in range(len(TARGET_CLASSES))},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    val_pred_file = pred_dir / "risk_lightgbm_validation.jsonl"
    test_pred_file = pred_dir / "risk_lightgbm_test.jsonl"
    save_predictions_jsonl(val_pred_file, df_val, val_probs, val_preds)
    save_predictions_jsonl(test_pred_file, df_test, test_probs, test_preds)

    print("============================================================")
    print("SAVED ARTIFACTS")
    print("============================================================")
    print(f"Model booster:      {model_dir / 'model.txt'}")
    print(f"Model config:       {model_dir / 'config.json'}")
    print(f"Model metrics:      {model_dir / 'metrics.json'}")
    print(f"Validation preds:   {val_pred_file}")
    print(f"Test preds:         {test_pred_file}")
    print("STATUS: COMPLETE")
    print("============================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustLens T-8 LightGBM Risk Classifier")
    parser.add_argument("--data-dir", default="data/processed/features", help="Path to feature dataset directory")
    parser.add_argument("--output-dir", default="artifacts", help="Path to artifacts root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-leaves", type=int, default=31, help="Max number of leaves in one tree")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of boosting iterations")
    parser.add_argument("--max-depth", type=int, default=-1, help="Maximum tree depth (-1 for unlimited)")
    parser.add_argument("--early-stopping-rounds", type=int, default=20, help="Early stopping patience rounds")
    return parser.parse_args()


def main():
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
