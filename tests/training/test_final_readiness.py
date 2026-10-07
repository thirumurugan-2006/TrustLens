"""
TrustLens Final Training Readiness Audit Tests (Phase 6E).
Validates dataset structural integrity, 4-class risk coverage, 4-relation evidence coverage,
retrieval, graph, and feature dataset presence, zero cross-split/target leakage,
and task-level training readiness decisions.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from app.training.readiness_audit import TrainingReadinessAuditor
from app.training.evaluation import EvaluationFramework


@pytest.fixture
def readiness_report():
    auditor = TrainingReadinessAuditor(data_dir="data/trustlens", processed_dir="data/processed")
    return auditor.run_audit()


# 1. Dataset version consistency
def test_dataset_version_consistency(readiness_report):
    assert readiness_report["dataset_version"] == "v0.2.0"
    manifest_path = Path("data/trustlens/dataset_manifest.json")
    assert manifest_path.is_file()
    manif_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manif_data["dataset_version"] == "v0.2.0"


# 2. Train/validation/test existence
def test_train_val_test_existence(readiness_report):
    counts = readiness_report["record_counts"]
    assert counts["total_posts"] == 1560
    assert counts["train"] == 1092
    assert counts["validation"] == 234
    assert counts["test"] == 234


# 3. Label completeness
def test_label_completeness(readiness_report):
    claim_detect = readiness_report["distributions"]["claim_detection"]
    assert "CLAIM" in claim_detect
    assert "NON_CLAIM" in claim_detect
    assert claim_detect["CLAIM"] > 0
    assert claim_detect["NON_CLAIM"] > 0


# 4. Risk-class presence (all 4 classes: HIGH, MEDIUM, LOW, INSUFFICIENT)
def test_risk_class_presence(readiness_report):
    risk_dist = readiness_report["distributions"]["risk"]
    assert risk_dist["HIGH"] == 990
    assert risk_dist["MEDIUM"] == 240
    assert risk_dist["LOW"] == 180
    assert risk_dist["INSUFFICIENT"] == 150
    # Confirm MEDIUM deficit is eliminated
    assert risk_dist["MEDIUM"] > 0


# 5. Evidence relation presence (all 4 classes: CONTRADICTS, SUPPORTS, NEUTRAL, INSUFFICIENT)
def test_evidence_relation_presence(readiness_report):
    ev_dist = readiness_report["distributions"]["evidence_relations"]
    assert ev_dist["CONTRADICTS"] == 1020
    assert ev_dist["SUPPORTS"] == 180
    assert ev_dist["NEUTRAL"] == 180
    assert ev_dist["INSUFFICIENT"] == 180
    # Confirm NEUTRAL deficit is eliminated
    assert ev_dist["NEUTRAL"] > 0


# 6. Retrieval dataset availability (Phase 6B)
def test_retrieval_dataset_availability():
    ret_dir = Path("data/processed/retrieval")
    manifest_file = ret_dir / "dataset_manifest.json"
    assert manifest_file.is_file()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["query_count"] == 1200
    assert data["pair_count"] == 4800
    assert data["negative_count"] == 3600
    assert (ret_dir / "eval_test_pool.jsonl").is_file()


# 7. Graph dataset availability (Phase 6C)
def test_graph_dataset_availability():
    graph_dir = Path("data/processed/graphs")
    manifest_file = graph_dir / "dataset_manifest.json"
    assert manifest_file.is_file()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["graph_count"] == 1560
    assert data["node_count"] == 7800
    assert data["edge_count"] == 6240


# 8. Feature dataset availability (Phase 6D)
def test_feature_dataset_availability():
    feat_dir = Path("data/processed/features")
    manifest_file = feat_dir / "feature_manifest.json"
    assert manifest_file.is_file()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["total_rows"] == 1560
    assert data["train_rows"] == 1092
    assert data["validation_rows"] == 234
    assert data["test_rows"] == 234
    assert data["feature_count"] == 92
    assert (feat_dir / "train.csv").is_file()
    assert (feat_dir / "validation.csv").is_file()
    assert (feat_dir / "test.csv").is_file()


# 9. Cross-split leakage (must be 0)
def test_cross_split_leakage(readiness_report):
    assert readiness_report["integrity_checks"]["cross_split_leakage_violations"] == 0
    assert readiness_report["leakage"]["cross_split_leakage"] == 0


# 10. Target leakage (must be 0)
def test_target_leakage():
    leak_file = Path("data/processed/features/feature_leakage_report.json")
    assert leak_file.is_file()
    data = json.loads(leak_file.read_text(encoding="utf-8"))
    assert data["leakage_detected"] is False
    assert len(data["forbidden_columns_found"]) == 0


# 11. Schema consistency across splits
def test_schema_consistency():
    feat_dir = Path("data/processed/features")
    df_tr = pd.read_csv(feat_dir / "train.csv")
    df_va = pd.read_csv(feat_dir / "validation.csv")
    df_te = pd.read_csv(feat_dir / "test.csv")
    assert list(df_tr.columns) == list(df_va.columns)
    assert list(df_tr.columns) == list(df_te.columns)


# 12. Readiness status generation
def test_readiness_status_generation(readiness_report):
    assert readiness_report["overall_status"] == "TRAINING_READY"
    assert readiness_report["training_ready"] is True
    assert len(readiness_report["blockers"]) == 0

    tasks = readiness_report["tasks"]
    assert tasks["T1"]["status"] == "READY"
    assert tasks["T2"]["status"] == "READY_WITH_LIMITATIONS"
    assert tasks["T3"]["status"] == "READY_WITH_LIMITATIONS"
    assert tasks["T4"]["status"] == "READY_WITH_LIMITATIONS"
    assert tasks["T5"]["status"] == "READY"
    assert tasks["T6"]["status"] == "READY"
    assert tasks["T7"]["status"] == "READY_WITH_LIMITATIONS"
    assert tasks["T8"]["status"] == "READY"


# 13. Calibration dependency (must be CALIBRATION_NOT_YET_AVAILABLE)
def test_calibration_dependency(readiness_report):
    calib = readiness_report["tasks"]["confidence_calibration"]
    assert calib["status"] == "CALIBRATION_NOT_YET_AVAILABLE"
    assert "predictions do not exist" in calib["notes"].lower()


# 14. Abstention dependency (must be READY_FOR_MODEL_PHASE)
def test_abstention_dependency(readiness_report):
    abst = readiness_report["tasks"]["selective_prediction_abstention"]
    assert abst["status"] == "READY_FOR_MODEL_PHASE"
    assert "evaluation functions prepared" in abst["notes"].lower()
