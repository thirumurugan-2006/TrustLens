import json
from pathlib import Path

import pytest

from app.dataset.clusterer import DatasetClusterer
from app.dataset.deduplicator import DatasetDeduplicator
from app.dataset.manifest import Phase4DatasetManifest
from app.dataset.provenance import ProvenanceManager
from app.dataset.source_registry import SourceRegistry
from app.dataset.splitter import DatasetSplitter
from app.training.readiness_audit import TrainingReadinessAuditor
from app.training.schemas import (
    EvidenceRelationLabel,
    ProvenanceSourceType,
    RiskLevel,
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingRisk,
)


@pytest.fixture
def trustlens_data_dir():
    return Path("data/trustlens")


# ---------------------------------------------------------------------------
# 1. New Source Registration
# ---------------------------------------------------------------------------
def test_new_source_registration(trustlens_data_dir):
    reg_path = trustlens_data_dir / "source_registry.json"
    assert reg_path.is_file(), "source_registry.json missing"
    registry = SourceRegistry(reg_path)

    # Approved remediation sources
    assert registry.is_permitted("Ministry of Corporate Affairs Public Registry & Corporate Disclosure Archive")
    assert registry.is_permitted("Public Consumer Complaints & Dispute Bulletin")

    # Quarantined source with LICENSE_REVIEW_REQUIRED
    pending_source = registry.get_source("src_pending_unreviewed")
    assert pending_source is not None
    assert pending_source.permission_status == "LICENSE_REVIEW_REQUIRED"
    assert not registry.is_permitted("Unverified Third Party Scraping Feed")


# ---------------------------------------------------------------------------
# 2. Provenance Validation
# ---------------------------------------------------------------------------
def test_provenance_validation(trustlens_data_dir):
    posts_path = trustlens_data_dir / "posts.jsonl"
    reg_path = trustlens_data_dir / "source_registry.json"
    registry = SourceRegistry(reg_path)

    count = 0
    with posts_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            post = TrainingPost.model_validate(json.loads(line.strip()))
            count += 1
            assert post.provenance is not None
            assert post.provenance.source_name
            assert post.provenance.license
            assert registry.is_permitted(post.provenance.source_name)
    assert count == 1560, f"Expected 1,560 posts, found {count}"


# ---------------------------------------------------------------------------
# 3. MEDIUM Label Validation
# ---------------------------------------------------------------------------
def test_medium_label_validation(trustlens_data_dir):
    risks_path = trustlens_data_dir / "annotated" / "risks.jsonl"
    medium_risks = []
    with risks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = TrainingRisk.model_validate(json.loads(line.strip()))
            if r.risk_level == RiskLevel.MEDIUM:
                medium_risks.append(r)

    assert len(medium_risks) == 240, f"Expected 240 MEDIUM risks, found {len(medium_risks)}"
    for r in medium_risks:
        assert len(r.risk_factors) > 0, "MEDIUM risk must specify observable risk factors"
        assert r.evidence_summary, "MEDIUM risk must document rationale"
        assert "guaranteed_return" not in r.risk_factors or "regulatory_ban" not in r.evidence_summary


# ---------------------------------------------------------------------------
# 4. NEUTRAL Evidence Validation
# ---------------------------------------------------------------------------
def test_neutral_evidence_validation(trustlens_data_dir):
    ev_path = trustlens_data_dir / "annotated" / "evidence.jsonl"
    neutral_ev = []
    with ev_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = TrainingEvidence.model_validate(json.loads(line.strip()))
            if e.relation_label == EvidenceRelationLabel.NEUTRAL:
                neutral_ev.append(e)

    assert len(neutral_ev) == 180, f"Expected 180 NEUTRAL evidence records, found {len(neutral_ev)}"
    for e in neutral_ev:
        assert e.evidence_text, "NEUTRAL evidence must have text"
        assert e.relation_label != EvidenceRelationLabel.INSUFFICIENT, "NEUTRAL must not be INSUFFICIENT"


# ---------------------------------------------------------------------------
# 5. Multilingual MEDIUM Records
# ---------------------------------------------------------------------------
def test_multilingual_medium_records(trustlens_data_dir):
    audit_file = trustlens_data_dir / "audit" / "medium_language_distribution.json"
    assert audit_file.is_file(), "medium_language_distribution.json missing"
    dist = json.loads(audit_file.read_text(encoding="utf-8"))

    for lang in ["en", "ta", "hi", "ta-en", "hi-en"]:
        assert lang in dist, f"Language {lang} missing from MEDIUM distribution"
        assert dist[lang] >= 40, f"Language {lang} has insufficient MEDIUM records: {dist[lang]}"


# ---------------------------------------------------------------------------
# 6. Evidence Relation Validation
# ---------------------------------------------------------------------------
def test_evidence_relation_validation(trustlens_data_dir):
    audit_file = trustlens_data_dir / "audit" / "evidence_relation_distribution.json"
    assert audit_file.is_file()
    dist = json.loads(audit_file.read_text(encoding="utf-8"))

    assert dist.get("SUPPORTS", 0) >= 180
    assert dist.get("CONTRADICTS", 0) >= 1000
    assert dist.get("NEUTRAL", 0) == 180
    assert dist.get("INSUFFICIENT", 0) >= 180


# ---------------------------------------------------------------------------
# 7. Double-Annotation Coverage
# ---------------------------------------------------------------------------
def test_double_annotation(trustlens_data_dir):
    stats_file = trustlens_data_dir / "audit" / "phase_6a_statistics.json"
    assert stats_file.is_file()
    stats = json.loads(stats_file.read_text(encoding="utf-8"))

    # Combined double-annotated count must be at least 262 (172 from v0.1.0 + 90 from Phase 6A)
    assert stats["double_annotated_count"] >= 260
    assert stats["new_records_added"] == 360


# ---------------------------------------------------------------------------
# 8. Adjudication Tracking
# ---------------------------------------------------------------------------
def test_adjudication(trustlens_data_dir):
    stats_file = trustlens_data_dir / "audit" / "phase_6a_statistics.json"
    stats = json.loads(stats_file.read_text(encoding="utf-8"))

    assert stats["adjudicated_count"] >= 260
    samples = stats.get("adjudication_sample_records", [])
    assert len(samples) > 0
    for s in samples:
        assert "annotator_A" in s
        assert "annotator_B" in s
        assert "adjudicator" in s
        assert "adjudicated_label" in s
        assert "adjudication_rationale" in s


# ---------------------------------------------------------------------------
# 9. Duplicate Detection
# ---------------------------------------------------------------------------
def test_duplicate_detection(trustlens_data_dir):
    dedup = DatasetDeduplicator()
    posts_path = trustlens_data_dir / "posts.jsonl"
    with posts_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        sample_post = TrainingPost.model_validate(json.loads(first_line))

    d_info = dedup.check_and_register(sample_post)
    assert not d_info.is_duplicate

    # Duplicate check on identical text
    d_info_dup = dedup.check_and_register(sample_post)
    assert d_info_dup.is_duplicate


# ---------------------------------------------------------------------------
# 10. Cluster Assignment
# ---------------------------------------------------------------------------
def test_cluster_assignment(trustlens_data_dir):
    clusterer = DatasetClusterer()
    posts_path = trustlens_data_dir / "posts.jsonl"
    posts = []
    with posts_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 12:
                break
            p = TrainingPost.model_validate(json.loads(line.strip()))
            posts.append(p)
            camp_id = p.metadata.get("campaign_id")
            clusterer.register_post(p, campaign_group_id=camp_id)

    clusters = clusterer.get_all_clusters()
    assert len(clusters) > 0


# ---------------------------------------------------------------------------
# 11. Leakage Prevention
# ---------------------------------------------------------------------------
def test_leakage_prevention(trustlens_data_dir):
    auditor = TrainingReadinessAuditor(data_dir=str(trustlens_data_dir))
    report = auditor.run_audit()

    assert report["integrity_checks"]["cross_split_leakage_violations"] == 0
    assert report["integrity_checks"]["synthetic_in_test_violations"] == 0


# ---------------------------------------------------------------------------
# 12. Dataset Versioning
# ---------------------------------------------------------------------------
def test_dataset_versioning(trustlens_data_dir):
    # Archive v0.1.0 exists and is preserved
    archive_dir = trustlens_data_dir / "archive" / "v0.1.0"
    assert archive_dir.is_dir(), "v0.1.0 archive directory must exist"
    assert (archive_dir / "dataset_manifest.json").is_file()

    old_manifest = json.loads((archive_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert old_manifest["dataset_version"] == "v0.1.0"
    assert old_manifest["total_records"] == 1200

    # Active manifest is v0.2.0
    new_manifest = json.loads((trustlens_data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert new_manifest["dataset_version"] == "v0.2.0"
    assert new_manifest["parent_version"] == "v0.1.0"
    assert new_manifest["total_records"] == 1560


# ---------------------------------------------------------------------------
# 13. Manifest Regeneration
# ---------------------------------------------------------------------------
def test_manifest_regeneration(trustlens_data_dir):
    manif_file = trustlens_data_dir / "dataset_manifest.json"
    manifest = Phase4DatasetManifest.model_validate_json(manif_file.read_text(encoding="utf-8"))

    assert manifest.total_records == 1560
    assert manifest.train_count == 1092
    assert manifest.validation_count == 234
    assert manifest.test_count == 234
    assert manifest.medium_count == 240
    assert manifest.neutral_count == 180
    assert manifest.synthetic_test_count == 0


# ---------------------------------------------------------------------------
# 14. Risk Distribution Calculation
# ---------------------------------------------------------------------------
def test_risk_distribution_calculation(trustlens_data_dir):
    audit_file = trustlens_data_dir / "audit" / "medium_risk_audit.json"
    audit_data = json.loads(audit_file.read_text(encoding="utf-8"))

    assert audit_data["total_medium_records"] == 240
    assert audit_data["split_distribution"]["train"] > 0
    assert audit_data["split_distribution"]["validation"] > 0
    assert audit_data["split_distribution"]["test"] > 0


# ---------------------------------------------------------------------------
# 15. Evidence Distribution Calculation
# ---------------------------------------------------------------------------
def test_evidence_distribution_calculation(trustlens_data_dir):
    audit_file = trustlens_data_dir / "audit" / "neutral_evidence_audit.json"
    audit_data = json.loads(audit_file.read_text(encoding="utf-8"))

    assert audit_data["total_neutral_evidence"] == 180
    assert audit_data["split_distribution"]["train"] > 0
    assert audit_data["split_distribution"]["validation"] > 0
    assert audit_data["split_distribution"]["test"] > 0


# ---------------------------------------------------------------------------
# 16. Phase 5 Re-Audit Invocation
# ---------------------------------------------------------------------------
def test_phase_5_reaudit_invocation(trustlens_data_dir):
    auditor = TrainingReadinessAuditor(data_dir=str(trustlens_data_dir))
    report = auditor.run_audit()

    # MEDIUM risk blocker must NOT be present
    assert not any("CRITICAL_DEFICIT_MEDIUM_RISK" in b for b in report["blockers"])
    # HIGH class dominance warning must NOT be present (HIGH is 63.5% < 75%)
    assert not any("HIGH_CLASS_DOMINANCE" in w for w in report["warnings"])
    # NEUTRAL evidence warning must NOT be present
    assert not any("EVIDENCE_NEUTRAL_MISSING" in w for w in report["warnings"])

    # Model architecture prerequisite blockers must remain reported honestly
    assert any("BLOCKER_RETRIEVAL_PAIRS_MISSING" in b for b in report["blockers"])
    assert any("BLOCKER_GAT_GRAPH_DATASET_MISSING" in b for b in report["blockers"])
    assert any("BLOCKER_LIGHTGBM_FEATURES_MISSING" in b for b in report["blockers"])

    assert report["overall_status"] == "TRAINING_NOT_READY"
    assert report["training_ready"] is False
