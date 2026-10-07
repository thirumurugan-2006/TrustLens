import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.dataset.provenance import ProvenanceManager
from app.training.baselines import (
    BaselineFramework,
    MajorityClassBaseline,
    TfidfLinearSVMBaseline,
    TfidfLogisticRegressionBaseline,
)
from app.training.evaluation import EvaluationFramework
from app.training.readiness_audit import TrainingReadinessAuditor
from app.training.schemas import (
    ClaimDetectionLabel,
    ClaimType,
    EvidenceRelationLabel,
    LanguageMetadata,
    ProvenanceMetadata,
    ProvenanceSourceType,
    RiskLevel,
    SplitMetadata,
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingRisk,
)
from app.input.schemas import Platform, PostContent, PostType


@pytest.fixture
def mock_provenance():
    return ProvenanceManager.create_provenance(
        source_type=ProvenanceSourceType.PUBLIC_DATASET,
        source_name="SEBI Alerts Archive",
        source_id="sebi_001",
        license_str="OGDL",
    )


@pytest.fixture
def sample_audit_env(tmp_path, mock_provenance):
    """Creates a miniature dataset environment with known properties for deterministic testing."""
    data_dir = tmp_path / "data_env"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create Source Registry
    reg_data = [
        {
            "source_id": "src_sebi",
            "source_name": "SEBI Alerts Archive",
            "source_type": "PUBLIC_DATASET",
            "license": "OGDL",
            "permission_status": "APPROVED",
        }
    ]
    (data_dir / "source_registry.json").write_text(json.dumps(reg_data), encoding="utf-8")

    # 2. Create sample posts, claims, evidence, risks
    posts = []
    claims = []
    evidence = []
    risks = []

    languages = ["en", "ta", "hi", "ta-en", "hi-en"]
    splits = [SplitName.train, SplitName.validation, SplitName.test]

    for i in range(15):
        pid = f"test_post_{i:03d}"
        cid = f"test_claim_{i:03d}"
        eid = f"test_ev_{i:03d}"
        rid = f"test_risk_{i:03d}"

        lang = languages[i % len(languages)]
        sp = splits[i % len(splits)]
        # Intentionally make 12 HIGH and 3 LOW, 0 MEDIUM to verify audit detection
        r_level = RiskLevel.HIGH if i < 12 else RiskLevel.LOW

        p = TrainingPost(
            post_id=pid,
            platform=Platform.reddit,
            post_type=PostType.text,
            content=PostContent(text=f"Test sample text {i}"),
            timestamp="2026-10-07T10:00:00Z",
            language_info=LanguageMetadata(primary=lang, languages=[lang]),
            provenance=mock_provenance,
            split_info=SplitMetadata(split=sp, campaign_group_id=f"camp_{i}"),
            metadata={"category": "FINANCIAL"},
        )
        posts.append(p)

        c = TrainingClaim(
            claim_id=cid,
            post_id=pid,
            claim_text=f"Test claim {i}",
            claim_type=ClaimType.FINANCIAL,
            detection_label=ClaimDetectionLabel.CLAIM,
            provenance=mock_provenance,
        )
        claims.append(c)

        e = TrainingEvidence(
            evidence_id=eid,
            claim_id=cid,
            evidence_text=f"Test evidence {i}",
            relation_label=EvidenceRelationLabel.CONTRADICTS if r_level == RiskLevel.HIGH else EvidenceRelationLabel.SUPPORTS,
            provenance=mock_provenance,
        )
        evidence.append(e)

        r = TrainingRisk(
            risk_id=rid,
            post_id=pid,
            claim_ids=[cid],
            risk_level=r_level,
            provenance=mock_provenance,
        )
        risks.append(r)

    # Write to files
    with (data_dir / "posts.jsonl").open("w", encoding="utf-8") as f:
        for p in posts:
            f.write(json.dumps(p.model_dump()) + "\n")
    with (data_dir / "claims.jsonl").open("w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(c.model_dump()) + "\n")
    with (data_dir / "evidence.jsonl").open("w", encoding="utf-8") as f:
        for e in evidence:
            f.write(json.dumps(e.model_dump()) + "\n")
    with (data_dir / "risks.jsonl").open("w", encoding="utf-8") as f:
        for r in risks:
            f.write(json.dumps(r.model_dump()) + "\n")

    return data_dir


# ---------------------------------------------------------------------------
# 1. Dataset Structural Audit
# ---------------------------------------------------------------------------

def test_dataset_structural_audit(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    assert report["record_counts"]["total_posts"] == 15
    assert report["record_counts"]["train"] > 0
    assert report["record_counts"]["validation"] > 0
    assert report["record_counts"]["test"] > 0
    assert report["integrity_checks"]["cross_split_leakage_violations"] == 0
    assert report["integrity_checks"]["synthetic_in_test_violations"] == 0


# ---------------------------------------------------------------------------
# 2. Label Completeness
# ---------------------------------------------------------------------------

def test_label_completeness(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    claims_dist = report["distributions"]["claim_detection"]
    assert "CLAIM" in claims_dist
    assert claims_dist["CLAIM"] == 15

    evidence_dist = report["distributions"]["evidence_relations"]
    assert "CONTRADICTS" in evidence_dist
    assert "SUPPORTS" in evidence_dist


# ---------------------------------------------------------------------------
# 3. Risk Distribution Audit (MEDIUM=0 Detection)
# ---------------------------------------------------------------------------

def test_risk_distribution_audit(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    risk_dist = report["distributions"]["risk"]
    assert risk_dist["MEDIUM"] == 0
    assert risk_dist["HIGH"] == 12

    # Auditor must flag MEDIUM deficit as a blocker
    assert any("CRITICAL_DEFICIT_MEDIUM_RISK" in b for b in report["blockers"])
    assert any("HIGH_CLASS_DOMINANCE" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# 4. Language x Risk Matrix
# ---------------------------------------------------------------------------

def test_language_x_risk_matrix(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    matrix = report["matrices"]["language_x_risk"]
    assert "en" in matrix
    assert "ta" in matrix
    assert "hi" in matrix
    assert matrix["en"]["MEDIUM"] == 0


# ---------------------------------------------------------------------------
# 5. Domain x Risk Matrix
# ---------------------------------------------------------------------------

def test_domain_x_risk_matrix(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    matrix = report["matrices"]["domain_x_risk"]
    assert "FINANCIAL" in matrix
    assert matrix["FINANCIAL"]["HIGH"] == 12
    assert matrix["FINANCIAL"]["LOW"] == 3


# ---------------------------------------------------------------------------
# 6. Split x Risk Matrix
# ---------------------------------------------------------------------------

def test_split_x_risk_matrix(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    matrix = report["matrices"]["split_x_risk"]
    assert "train" in matrix
    assert "validation" in matrix
    assert "test" in matrix
    assert matrix["test"]["MEDIUM"] == 0


# ---------------------------------------------------------------------------
# 7. Split x Language Matrix
# ---------------------------------------------------------------------------

def test_split_x_language_matrix(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    matrix = report["matrices"]["split_x_language"]
    assert "train" in matrix
    assert "validation" in matrix
    assert "test" in matrix


# ---------------------------------------------------------------------------
# 8. Provenance Audit
# ---------------------------------------------------------------------------

def test_provenance_audit(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    assert report["integrity_checks"]["missing_provenance_count"] == 0
    assert report["integrity_checks"]["unapproved_sources_count"] == 0


# ---------------------------------------------------------------------------
# 9. Leakage Audit
# ---------------------------------------------------------------------------

def test_leakage_audit(sample_audit_env, mock_provenance):
    # Introduce cross-split leakage into the environment
    leaking_post = TrainingPost(
        post_id="leak_post_999",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Leaking campaign post"),
        timestamp="2026-10-07T10:00:00Z",
        provenance=mock_provenance,
        # Shared campaign camp_0 placed in test split while test_post_000 is in train split!
        split_info=SplitMetadata(split=SplitName.test, campaign_group_id="camp_0"),
    )
    with (sample_audit_env / "posts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(leaking_post.model_dump()) + "\n")

    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    assert report["integrity_checks"]["cross_split_leakage_violations"] > 0
    assert any("CROSS_SPLIT_LEAKAGE" in b for b in report["blockers"])


# ---------------------------------------------------------------------------
# 10. Synthetic Exclusion
# ---------------------------------------------------------------------------

def test_synthetic_exclusion(sample_audit_env):
    prov_synth = ProvenanceManager.create_provenance(
        source_type=ProvenanceSourceType.SYNTHETIC,
        source_name="SyntheticGen",
    )
    synth_in_test = TrainingPost(
        post_id="synth_in_test_01",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Synthetic post in test"),
        timestamp="2026-10-07T10:00:00Z",
        provenance=prov_synth,
        is_example=True,
        split_info=SplitMetadata(split=SplitName.test),
    )
    with (sample_audit_env / "posts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(synth_in_test.model_dump()) + "\n")

    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    assert report["integrity_checks"]["synthetic_in_test_violations"] > 0
    assert any("SYNTHETIC_TEST_VIOLATION" in b for b in report["blockers"])


# ---------------------------------------------------------------------------
# 11. Annotation Quality
# ---------------------------------------------------------------------------

def test_annotation_quality_audit(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    assert "annotation_audit" in report
    assert "double_annotated_count" in report["annotation_audit"]


# ---------------------------------------------------------------------------
# 12. Readiness Decision Logic
# ---------------------------------------------------------------------------

def test_readiness_decision_logic(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    # Because blockers exist (MEDIUM=0, missing retrieval/GAT datasets), status must be TRAINING_NOT_READY
    assert report["overall_status"] == "TRAINING_NOT_READY"
    assert report["training_ready"] is False
    assert len(report["blockers"]) > 0


# ---------------------------------------------------------------------------
# 13. Evaluation Metrics Framework
# ---------------------------------------------------------------------------

def test_evaluation_metrics():
    # Classification metrics
    y_true = ["HIGH", "LOW", "HIGH", "INSUFFICIENT"]
    y_pred = ["HIGH", "LOW", "INSUFFICIENT", "INSUFFICIENT"]
    metrics = EvaluationFramework.calculate_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["accuracy"] == 0.75

    # Retrieval metrics
    retrieved = [["doc1", "doc2", "doc3"], ["doc4", "doc5", "doc6"]]
    relevant = [{"doc2"}, {"doc4"}]
    r_metrics = EvaluationFramework.calculate_retrieval_metrics(retrieved, relevant, k_values=[1, 3])
    assert "mrr" in r_metrics
    assert "recall@1" in r_metrics
    assert r_metrics["mrr"] == 0.75

    # Calibration metrics
    y_true_bin = [1, 0, 1, 1]
    y_prob = [0.9, 0.1, 0.8, 0.6]
    cal_metrics = EvaluationFramework.calculate_calibration_metrics(y_true_bin, y_prob)
    assert "ece" in cal_metrics
    assert "brier_score" in cal_metrics

    # Abstention metrics
    confidences = [0.95, 0.40, 0.85, 0.30]
    abst_metrics = EvaluationFramework.calculate_abstention_metrics(y_true, y_pred, confidences, threshold=0.5)
    assert "coverage" in abst_metrics
    assert "selective_accuracy" in abst_metrics

    # OCR metrics
    ocr_metrics = EvaluationFramework.calculate_ocr_metrics(["abc"], ["abd"])
    assert "cer" in ocr_metrics
    assert round(ocr_metrics["cer"], 2) == 0.33


# ---------------------------------------------------------------------------
# 14. Baseline Configuration
# ---------------------------------------------------------------------------

def test_baseline_configuration():
    # Majority Class Baseline
    maj = BaselineFramework.get_baseline("majority")
    maj.fit(["HIGH", "HIGH", "LOW"])
    assert maj.predict(["any_text"])[0] == "HIGH"

    # TF-IDF Logistic Regression Baseline
    lr = BaselineFramework.get_baseline("tfidf_lr")
    assert isinstance(lr, TfidfLogisticRegressionBaseline)

    # TF-IDF Linear SVM Baseline
    svm = BaselineFramework.get_baseline("tfidf_svm")
    assert isinstance(svm, TfidfLinearSVMBaseline)


# ---------------------------------------------------------------------------
# 15. Task Readiness Mapping
# ---------------------------------------------------------------------------

def test_task_readiness_mapping(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    tasks = report["tasks"]
    assert tasks["claim_detection"]["status"] == "READY"
    assert tasks["retrieval_ranking"]["status"] == "BLOCKED"
    assert tasks["gat_relational_graph"]["status"] == "BLOCKED"
    assert tasks["lightgbm_risk_classifier"]["status"] in ("NOT_READY", "BLOCKED")


# ---------------------------------------------------------------------------
# 16. JSON Report Generation
# ---------------------------------------------------------------------------

def test_json_report_generation(sample_audit_env):
    auditor = TrainingReadinessAuditor(data_dir=str(sample_audit_env))
    report = auditor.run_audit()

    report_file = sample_audit_env / "training_readiness_report.json"
    assert report_file.is_file()
    saved_data = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved_data["overall_status"] == report["overall_status"]
    assert saved_data["record_counts"]["total_posts"] == 15


# ---------------------------------------------------------------------------
# 17. CLI Exit Codes
# ---------------------------------------------------------------------------

def test_cli_exit_codes(sample_audit_env):
    # Run CLI using subprocess
    res = subprocess.run(
        [sys.executable, "-m", "app.training.readiness_audit", "--data-dir", str(sample_audit_env)],
        capture_output=True,
        text=True,
    )
    # Must exit with code 1 due to blockers present
    assert res.returncode == 1
    assert "TRUSTLENS TRAINING READINESS AUDIT" in res.stdout
    assert "Overall Status:" in res.stdout
