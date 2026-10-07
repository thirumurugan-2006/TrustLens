import json
from pathlib import Path

import pytest

from app.claims.schemas import SourceSpan
from app.dataset.annotation_queue import AnnotationQueue, AnnotationQueueError
from app.dataset.builder import DevelopmentDatasetBuilder
from app.dataset.clusterer import DatasetClusterer
from app.dataset.deduplicator import DatasetDeduplicator
from app.dataset.manifest import DatasetManifestBuilder, Phase4DatasetManifest
from app.dataset.normalizer import DatasetNormalizer
from app.dataset.provenance import ProvenanceError, ProvenanceManager
from app.dataset.quality_control import DatasetQualityControl
from app.dataset.schemas import (
    AnnotationTaskStatus,
    DuplicateType,
    HumanAnnotationSubmission,
    RawDataRecord,
)
from app.dataset.source_registry import SourceEntry, SourceRegistry
from app.dataset.splitter import DatasetSplitter
from app.dataset.validator import DatasetValidator
from app.input.schemas import AuthorInfo, Platform, PostContent, PostMedia, PostType
from app.training.schemas import (
    AnnotationConfidence,
    AnnotationMetadata,
    ClaimDetectionLabel,
    ClaimType,
    EvidenceRelationLabel,
    LanguageMetadata,
    ProvenanceMetadata,
    ProvenanceSourceType,
    ReviewStatus,
    RiskLevel,
    SplitMetadata,
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingRisk,
)
from app.validation.post_validator import validate_post


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_registry_path(tmp_path):
    reg_file = tmp_path / "source_registry.json"
    registry = SourceRegistry(registry_path=reg_file)
    registry.register_source(
        SourceEntry(
            source_id="src_sebi_test",
            source_name="SEBI Alerts Archive",
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
            license="Open Government Data License (OGDL)",
            permission_status="APPROVED",
            collection_method="regulatory_bulletin",
            language=["en", "hi"],
            category=["financial"],
        )
    )
    registry.register_source(
        SourceEntry(
            source_id="src_unreviewed_test",
            source_name="Unreviewed Third-Party Feed",
            source_type=ProvenanceSourceType.USER_PROVIDED,
            license="PENDING_REVIEW",
            permission_status="LICENSE_REVIEW_REQUIRED",
            collection_method="scraping_submission",
            language=["en"],
            category=["general"],
        )
    )
    registry.save()
    return reg_file


@pytest.fixture
def real_provenance():
    return ProvenanceManager.create_provenance(
        source_type=ProvenanceSourceType.PUBLIC_DATASET,
        source_name="SEBI Alerts Archive",
        source_id="sebi_2026_case_01",
        license_str="Open Government Data License (OGDL)",
        collection_method="curated_official_registry",
    )


# ---------------------------------------------------------------------------
# 1. Source Registry
# ---------------------------------------------------------------------------

def test_source_registry_registration_and_loading(sample_registry_path):
    registry = SourceRegistry(sample_registry_path)
    src_sebi = registry.get_source("src_sebi_test")
    assert src_sebi is not None
    assert src_sebi.source_name == "SEBI Alerts Archive"
    assert src_sebi.source_type == ProvenanceSourceType.PUBLIC_DATASET
    assert src_sebi.permission_status == "APPROVED"


# ---------------------------------------------------------------------------
# 2. License Validation
# ---------------------------------------------------------------------------

def test_source_license_validation(sample_registry_path):
    registry = SourceRegistry(sample_registry_path)
    # Approved source
    assert registry.is_permitted("src_sebi_test") is True
    assert registry.validate_source_license("SEBI Alerts Archive") is True

    # Unreviewed / pending source
    assert registry.is_permitted("src_unreviewed_test") is False
    assert registry.validate_source_license("Unreviewed Third-Party Feed") is False


# ---------------------------------------------------------------------------
# 3. Real Record Ingestion
# ---------------------------------------------------------------------------

def test_real_record_ingestion(tmp_path, real_provenance):
    normalizer = DatasetNormalizer()
    jsonl_file = tmp_path / "real_input.jsonl"
    record_payload = {
        "post_id": "real_post_001",
        "text": "SEBI Alert: Unregistered entity promising 40% monthly returns via Telegram link.",
        "platform": "reddit",
        "author": {"username": "consumer_advocate_01", "verified": True},
        "timestamp": "2026-10-07T10:00:00Z",
    }
    jsonl_file.write_text(json.dumps(record_payload) + "\n", encoding="utf-8")

    parsed_raw = normalizer.parse_file(jsonl_file)
    assert len(parsed_raw) == 1

    post = normalizer.normalize_record(parsed_raw[0], real_provenance)
    assert post.post_id == "real_post_001"
    assert "SEBI Alert" in post.content.text
    assert post.provenance.source_name == "SEBI Alerts Archive"
    assert post.author.verified is True


# ---------------------------------------------------------------------------
# 4. Provenance Preservation
# ---------------------------------------------------------------------------

def test_provenance_preservation(real_provenance):
    ProvenanceManager.validate_provenance(real_provenance)
    assert real_provenance.source_type == ProvenanceSourceType.PUBLIC_DATASET
    assert real_provenance.license == "Open Government Data License (OGDL)"
    assert real_provenance.collection_method == "curated_official_registry"
    assert real_provenance.source_id == "sebi_2026_case_01"


# ---------------------------------------------------------------------------
# 5. Multilingual Metadata
# ---------------------------------------------------------------------------

def test_multilingual_metadata(real_provenance):
    # Pure Tamil
    post_ta = TrainingPost(
        post_id="post_ta_001",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="தினமும் ₹2,000 சம்பாதிக்க பகுதி நேர வேலை."),
        timestamp="2026-10-07T10:00:00Z",
        language_info=LanguageMetadata(
            primary="ta",
            languages=["ta"],
            script=["Tamil"],
            code_mixed=False,
            transliterated=False,
        ),
        provenance=real_provenance,
    )
    assert post_ta.language_info.primary == "ta"
    assert "Tamil" in post_ta.language_info.script

    # Hinglish (Code-mixed)
    post_hinglish = TrainingPost(
        post_id="post_hi_en_001",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Daily ₹3,000 earn karne ke liye telegram channel join karein."),
        timestamp="2026-10-07T10:00:00Z",
        language_info=LanguageMetadata(
            primary="hi-en",
            languages=["hi", "en"],
            script=["Latin"],
            code_mixed=True,
            transliterated=True,
        ),
        provenance=real_provenance,
    )
    assert post_hinglish.language_info.code_mixed is True
    assert post_hinglish.language_info.transliterated is True


# ---------------------------------------------------------------------------
# 6. Translation Grouping
# ---------------------------------------------------------------------------

def test_translation_grouping(real_provenance):
    clusterer = DatasetClusterer()

    post_en = TrainingPost(
        post_id="post_en_trans",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Earn guaranteed 30% weekly return on crypto deposit."),
        timestamp="2026-10-07T10:00:00Z",
        provenance=real_provenance,
    )
    post_ta = TrainingPost(
        post_id="post_ta_trans",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="கிரிப்டோ முதலீட்டில் வாரம் 30% உறுதி செய்யப்பட்ட லாபம் பெறுங்கள்."),
        timestamp="2026-10-07T10:00:00Z",
        provenance=real_provenance,
    )

    clusterer.register_post(post_en, translation_group_id="trans_scam_campaign_101")
    clusterer.register_post(post_ta, translation_group_id="trans_scam_campaign_101")

    assert clusterer.get_composite_cluster("post_en_trans") == clusterer.get_composite_cluster("post_ta_trans")


# ---------------------------------------------------------------------------
# 7. Annotation Workflow
# ---------------------------------------------------------------------------

def test_annotation_workflow(real_provenance):
    queue = AnnotationQueue()
    post = TrainingPost(
        post_id="post_ann_flow",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Send ₹1,000 registration fee to get verified part-time job."),
        timestamp="2026-10-07T10:00:00Z",
        provenance=real_provenance,
    )

    task = queue.enqueue(post)
    assert task.status == AnnotationTaskStatus.PENDING

    queue.assign(task.annotation_task_id, annotator_id="annotator_primary")
    assert task.status == AnnotationTaskStatus.IN_PROGRESS

    submission = HumanAnnotationSubmission(
        annotator_id="annotator_primary",
        claim_detection=ClaimDetectionLabel.CLAIM,
        claim_type=ClaimType.JOB,
        risk_level=RiskLevel.HIGH,
        confidence=AnnotationConfidence.HIGH,
        notes="Upfront payment requested for job offer.",
    )
    queue.submit_annotation(task.annotation_task_id, submission)
    assert task.status == AnnotationTaskStatus.SUBMITTED

    accepted = queue.accept(task.annotation_task_id)
    assert accepted.status == AnnotationTaskStatus.ACCEPTED
    assert accepted.consensus_submission.risk_level == RiskLevel.HIGH


# ---------------------------------------------------------------------------
# 8. Human Label Validation
# ---------------------------------------------------------------------------

def test_human_label_validation(real_provenance):
    claim = TrainingClaim(
        post_id="post_label_test",
        claim_text="Guaranteed ₹20,000 return within 15 days",
        claim_type=ClaimType.FINANCIAL,
        detection_label=ClaimDetectionLabel.CLAIM,
        check_worthiness=0.9,
        provenance=real_provenance,
    )
    qc_res = DatasetQualityControl.validate_claim(claim, existing_post_ids={"post_label_test"})
    assert qc_res.passed is True
    assert qc_res.status == "ACCEPTED"


# ---------------------------------------------------------------------------
# 9. Evidence Relation Validation (INSUFFICIENT != CONTRADICTS)
# ---------------------------------------------------------------------------

def test_evidence_relation_validation(real_provenance):
    # Contradicting evidence
    ev_contra = TrainingEvidence(
        claim_id="claim_001",
        evidence_text="SEBI caution notice confirms platform is completely unauthorized.",
        relation_label=EvidenceRelationLabel.CONTRADICTS,
        provenance=real_provenance,
    )
    assert ev_contra.relation_label == EvidenceRelationLabel.CONTRADICTS

    # Insufficient evidence
    ev_insuf = TrainingEvidence(
        claim_id="claim_002",
        evidence_text="No independent records found regarding stated venture.",
        relation_label=EvidenceRelationLabel.INSUFFICIENT,
        provenance=real_provenance,
    )
    assert ev_insuf.relation_label == EvidenceRelationLabel.INSUFFICIENT
    assert ev_insuf.relation_label != ev_contra.relation_label

    qc_res = DatasetQualityControl.validate_evidence(ev_insuf, existing_claim_ids={"claim_002"})
    assert qc_res.passed is True


# ---------------------------------------------------------------------------
# 10. Risk Validation
# ---------------------------------------------------------------------------

def test_risk_validation(real_provenance):
    risk = TrainingRisk(
        post_id="post_001",
        claim_ids=["claim_001"],
        risk_level=RiskLevel.HIGH,
        risk_factors=["guaranteed_return", "upfront_fee"],
        evidence_summary="The offer guarantees unrealistic returns contradicted by SEBI alerts.",
        confidence=AnnotationConfidence.HIGH,
        provenance=real_provenance,
    )
    qc_res = DatasetQualityControl.validate_risk(risk, existing_post_ids={"post_001"})
    assert qc_res.passed is True


# ---------------------------------------------------------------------------
# 11. QC Workflow
# ---------------------------------------------------------------------------

def test_qc_workflow(real_provenance):
    valid_post = TrainingPost(
        post_id="post_qc_ok",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Valid post text for quality control verification."),
        timestamp="2026-10-07T10:00:00Z",
        provenance=real_provenance,
    )
    assert DatasetQualityControl.validate_post(valid_post).passed is True

    # Invalid post missing canonical text
    invalid_post = valid_post.model_copy(deep=True)
    invalid_post.content.text = ""
    qc_bad = DatasetQualityControl.validate_post(invalid_post)
    assert qc_bad.passed is False
    assert any("MISSING_CANONICAL_TEXT" in r for r in qc_bad.reason_codes)


# ---------------------------------------------------------------------------
# 12. Multi-Annotator Disagreement
# ---------------------------------------------------------------------------

def test_multi_annotator_disagreement(real_provenance):
    queue = AnnotationQueue()
    post = TrainingPost(
        post_id="post_disagree",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Ambiguous discount offer on mobile app."),
        timestamp="2026-10-07T10:00:00Z",
        provenance=real_provenance,
    )
    task = queue.enqueue(post)

    sub_a = HumanAnnotationSubmission(
        annotator_id="annotator_A",
        claim_detection=ClaimDetectionLabel.CLAIM,
        claim_type=ClaimType.SHOPPING,
        risk_level=RiskLevel.HIGH,
    )
    sub_b = HumanAnnotationSubmission(
        annotator_id="annotator_B",
        claim_detection=ClaimDetectionLabel.CLAIM,
        claim_type=ClaimType.SHOPPING,
        risk_level=RiskLevel.MEDIUM,
    )

    task.submissions.extend([sub_a, sub_b])
    # Both annotations preserved without overwrite
    assert len(task.submissions) == 2
    assert task.submissions[0].risk_level == RiskLevel.HIGH
    assert task.submissions[1].risk_level == RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# 13. Adjudication
# ---------------------------------------------------------------------------

def test_adjudication_workflow(real_provenance):
    queue = AnnotationQueue()
    post = TrainingPost(
        post_id="post_adjudicate",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Disputed risk assessment text."),
        timestamp="2026-10-07T10:00:00Z",
        provenance=real_provenance,
    )
    task = queue.enqueue(post)
    sub_a = HumanAnnotationSubmission(annotator_id="annotator_A", risk_level=RiskLevel.HIGH)
    sub_b = HumanAnnotationSubmission(annotator_id="annotator_B", risk_level=RiskLevel.MEDIUM)
    task.submissions.extend([sub_a, sub_b])

    # Lead adjudicator settles on MEDIUM based on evidence
    consensus = HumanAnnotationSubmission(
        annotator_id="lead_adjudicator",
        risk_level=RiskLevel.MEDIUM,
        notes="Adjudicated to MEDIUM as evidence was partial but not conclusive.",
    )
    queue.accept(task.annotation_task_id, consensus_submission=consensus)
    assert task.status == AnnotationTaskStatus.ACCEPTED
    assert task.consensus_submission.annotator_id == "lead_adjudicator"
    assert task.consensus_submission.risk_level == RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# 14. Leakage Clustering
# ---------------------------------------------------------------------------

def test_leakage_clustering(real_provenance):
    clusterer = DatasetClusterer()
    p1 = TrainingPost(
        post_id="post_camp_01",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Send deposit to upi:fasttrade@okaxis to start earning."),
        timestamp="2026-10-07T10:00:00Z",
        provenance=real_provenance,
    )
    p2 = TrainingPost(
        post_id="post_camp_02",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Emergency activation fee needed at fasttrade@okaxis immediately."),
        timestamp="2026-10-07T10:00:00Z",
        provenance=real_provenance,
    )

    clusterer.register_post(p1)
    clusterer.register_post(p2)

    assert clusterer.get_composite_cluster("post_camp_01") == clusterer.get_composite_cluster("post_camp_02")


# ---------------------------------------------------------------------------
# 15. Cluster-Safe Split
# ---------------------------------------------------------------------------

def test_cluster_safe_split(real_provenance):
    splitter = DatasetSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    clusterer = DatasetClusterer()

    posts = []
    for i in range(12):
        p = TrainingPost(
            post_id=f"p_split_{i}",
            platform=Platform.reddit,
            post_type=PostType.text,
            content=PostContent(text=f"Sample post for splitting test {i}"),
            timestamp="2026-10-07T10:00:00Z",
            provenance=real_provenance,
        )
        posts.append(p)
        # First 4 in campaign A, next 4 in campaign B, next 4 in campaign C
        camp_id = f"camp_{i // 4}"
        clusterer.register_post(p, campaign_group_id=camp_id)

    split_map = splitter.split_posts(posts, clusterer)

    # All posts in campaign 0 must be in the exact same split
    assert split_map["p_split_0"] == split_map["p_split_1"] == split_map["p_split_2"] == split_map["p_split_3"]
    # All posts in campaign 1 must be in the exact same split
    assert split_map["p_split_4"] == split_map["p_split_5"] == split_map["p_split_6"] == split_map["p_split_7"]


# ---------------------------------------------------------------------------
# 16. Synthetic Exclusion from Test
# ---------------------------------------------------------------------------

def test_synthetic_exclusion_from_test():
    splitter = DatasetSplitter()
    clusterer = DatasetClusterer()

    prov_synth = ProvenanceManager.create_provenance(
        source_type=ProvenanceSourceType.SYNTHETIC,
        source_name="SyntheticGen",
    )
    prov_real = ProvenanceManager.create_provenance(
        source_type=ProvenanceSourceType.PUBLIC_DATASET,
        source_name="OfficialArchive",
    )

    synth_post = TrainingPost(
        post_id="synth_post_99",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Synthetic benchmark text"),
        timestamp="2026-10-07T10:00:00Z",
        provenance=prov_synth,
        is_example=True,
    )
    real_post = TrainingPost(
        post_id="real_post_99",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Real post observation text"),
        timestamp="2026-10-07T10:00:00Z",
        provenance=prov_real,
        is_example=False,
    )

    clusterer.register_post(synth_post)
    clusterer.register_post(real_post)

    split_map = splitter.split_posts([synth_post, real_post], clusterer)
    assert split_map["synth_post_99"] != SplitName.test


# ---------------------------------------------------------------------------
# 17. Manifest Generation
# ---------------------------------------------------------------------------

def test_manifest_generation(real_provenance, tmp_path):
    p = TrainingPost(
        post_id="post_manif",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Sample post for manifest generation."),
        timestamp="2026-10-07T10:00:00Z",
        provenance=real_provenance,
        split_info=SplitMetadata(split=SplitName.train),
    )
    manifest = DatasetManifestBuilder.build_manifest(
        posts=[p],
        dataset_id="trustlens_v0.1.0_dev",
        dataset_version="v0.1.0",
        double_annotated_count=1,
        adjudicated_count=1,
    )
    assert manifest.total_records == 1
    assert manifest.train_count == 1
    assert manifest.double_annotated_count == 1

    manif_file = tmp_path / "dataset_manifest.json"
    DatasetManifestBuilder.save_manifest(manifest, manif_file)
    assert manif_file.is_file()


# ---------------------------------------------------------------------------
# 18. Dataset Readiness Validation
# ---------------------------------------------------------------------------

def test_dataset_readiness_validation(real_provenance):
    validator = DatasetValidator()
    post = TrainingPost(
        post_id="post_ready",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Verified post meeting all quality gates."),
        timestamp="2026-10-07T10:00:00Z",
        provenance=real_provenance,
        split_info=SplitMetadata(split=SplitName.train),
    )
    claim = TrainingClaim(
        claim_id="claim_ready",
        post_id="post_ready",
        claim_text="Verified post meeting all quality gates.",
        claim_type=ClaimType.FINANCIAL,
        detection_label=ClaimDetectionLabel.CLAIM,
        provenance=real_provenance,
    )
    risk = TrainingRisk(
        risk_id="risk_ready",
        post_id="post_ready",
        claim_ids=["claim_ready"],
        risk_level=RiskLevel.HIGH,
        provenance=real_provenance,
    )

    report = validator.validate_dataset([post], claims=[claim], risks=[risk])
    assert report.valid is True
    assert report.total_records == 1
    assert report.schema_valid is True
    assert report.provenance_valid is True
    assert report.synthetic_in_test_violations == 0
    assert report.cross_split_leakage_violations == 0
