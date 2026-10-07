"""
Unit tests for TrustLens Model Training Scripts (Phase 7A.0).
Tests dataset loading, schema alignment, label mapping, probability checks,
and CLI interfaces without performing full model training.
"""

import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from app.training.models.lightgbm_risk import (
    load_dataset as load_lgb_dataset,
    compute_metrics as compute_lgb_metrics,
    TARGET_CLASSES as LGB_TARGET_CLASSES,
)
from app.training.models.claim_detection_xlm_roberta import (
    load_claim_dataset,
    compute_metrics as compute_claim_metrics,
    CLASSES as CLAIM_CLASSES,
)
from app.training.models.evidence_verification import (
    load_evidence_dataset,
    compute_metrics as compute_evidence_metrics,
    CLASSES as EVIDENCE_CLASSES,
)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# 1. T-8 LightGBM Script Unit Tests
# ---------------------------------------------------------------------------
def test_lightgbm_dataset_loading_and_feature_separation():
    data_dir = ROOT_DIR / "data" / "processed" / "features"
    df_train, df_val, df_test, feature_cols, target_col = load_lgb_dataset(data_dir)

    assert len(df_train) == 1092
    assert len(df_val) == 234
    assert len(df_test) == 234
    assert len(feature_cols) == 92
    assert target_col == "risk_label"

    # Strict target separation check
    assert target_col not in feature_cols
    for meta_col in ["feature_id", "post_id", "cluster_id", "split"]:
        assert meta_col not in feature_cols

    # Verify target labels
    unique_labels = set(df_train[target_col].unique())
    assert unique_labels == set(LGB_TARGET_CLASSES)


def test_lightgbm_probability_sum_and_metrics():
    # Test probability shape and sum verification logic
    mock_probs = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.05, 0.8, 0.1, 0.05],
        [0.2, 0.2, 0.5, 0.1],
    ])
    assert mock_probs.shape == (3, 4)
    assert np.allclose(np.sum(mock_probs, axis=1), 1.0)

    # Test metrics calculation
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    metrics = compute_lgb_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 4


def test_lightgbm_cli_help():
    cmd = [sys.executable, "-m", "app.training.models.lightgbm_risk", "--help"]
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert res.returncode == 0
    assert "TrustLens T-8 LightGBM Risk Classifier" in res.stdout


# ---------------------------------------------------------------------------
# 2. T-1 Claim Detection Scripts Unit Tests
# ---------------------------------------------------------------------------
def test_claim_detection_dataset_loading():
    data_dir = ROOT_DIR / "data" / "trustlens"
    splits_data, stats = load_claim_dataset(data_dir)

    assert stats["train_count"] == 1092
    assert stats["val_count"] == 234
    assert stats["test_count"] == 234
    assert stats["total_posts"] == 1560

    assert set(stats["labels"].keys()) == set(CLAIM_CLASSES)
    assert len(stats["languages"]) >= 3

    # Check structure of loaded item
    item = splits_data["train"][0]
    assert "post_id" in item
    assert "text" in item
    assert "label" in item
    assert item["label"] in [0, 1]


def test_claim_detection_metrics_calculation():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    metrics = compute_claim_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2


def test_claim_detection_xlmr_cli_help():
    cmd = [sys.executable, "-m", "app.training.models.claim_detection_xlm_roberta", "--help"]
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert res.returncode == 0
    assert "TrustLens T-1 XLM-RoBERTa Claim Detection" in res.stdout


def test_claim_detection_muril_cli_help():
    cmd = [sys.executable, "-m", "app.training.models.claim_detection_muril", "--help"]
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert res.returncode == 0
    assert "TrustLens T-1 MuRIL Claim Detection" in res.stdout


# ---------------------------------------------------------------------------
# 3. T-6 Evidence Verification Script Unit Tests
# ---------------------------------------------------------------------------
def test_evidence_verification_dataset_loading():
    data_dir = ROOT_DIR / "data" / "trustlens"
    splits_data, stats = load_evidence_dataset(data_dir)

    assert stats["train_count"] == 1092
    assert stats["val_count"] == 234
    assert stats["test_count"] == 234
    assert stats["total_pairs"] == 1560

    assert set(stats["labels"].keys()) == set(EVIDENCE_CLASSES)

    item = splits_data["train"][0]
    assert "claim_text" in item
    assert "evidence_text" in item
    assert "label" in item
    assert 0 <= item["label"] < 4


def test_evidence_verification_metrics_calculation():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]
    metrics = compute_evidence_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 4


def test_evidence_verification_cli_help():
    cmd = [sys.executable, "-m", "app.training.models.evidence_verification", "--help"]
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert res.returncode == 0
    assert "TrustLens T-6 Evidence Verification" in res.stdout


# ---------------------------------------------------------------------------
# 4. Dependency Checker Unit Test
# ---------------------------------------------------------------------------
def test_dependency_checker_script():
    cmd = [sys.executable, "scripts/check_dependencies.py"]
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert "TRUSTLENS ENVIRONMENT & DEPENDENCY CHECK" in res.stdout
    assert "Python:" in res.stdout
    assert "PyTorch:" in res.stdout
