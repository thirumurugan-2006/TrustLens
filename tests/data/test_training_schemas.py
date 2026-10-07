import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from app.training.schemas import (
    AnnotationConfidence,
    AnnotationMetadata,
    ClaimDetectionLabel,
    ClaimType,
    DatasetManifest,
    EvidenceRelationLabel,
    LanguageMetadata,
    ProvenanceMetadata,
    ProvenanceSourceType,
    QueryType,
    ReviewStatus,
    RiskLevel,
    SplitMetadata,
    SplitName,
    TrainingAtomicClaim,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingQuery,
    TrainingRisk,
)
from app.training.examples import get_synthetic_examples


def test_post_schema_valid_and_invalid():
    prov = ProvenanceMetadata(
        source_type=ProvenanceSourceType.TRUSTLENS_ANNOTATED,
        source_name="Manual Annotation Batch 1",
    )
    post = TrainingPost(
        post_id="post_test_01",
        platform="reddit",
        post_type="text",
        text="Invest ₹10,000 for guaranteed returns",
        provenance=prov,
    )
    assert post.post_id == "post_test_01"
    assert post.content.text == "Invest ₹10,000 for guaranteed returns"
    assert post.provenance.source_type == ProvenanceSourceType.TRUSTLENS_ANNOTATED

    # Missing provenance must fail validation
    with pytest.raises(ValidationError):
        TrainingPost(post_id="p2", platform="text")


def test_claim_schema_validation():
    prov = ProvenanceMetadata(
        source_type=ProvenanceSourceType.PUBLIC_DATASET,
        source_name="Benchmark Corpus",
    )
    claim = TrainingClaim(
        claim_id="c_01",
        post_id="p_01",
        claim_text="Double your investment in 7 days",
        claim_type=ClaimType.FINANCIAL,
        detection_label=ClaimDetectionLabel.CLAIM,
        provenance=prov,
    )
    assert claim.claim_type == ClaimType.FINANCIAL
    assert claim.detection_label == ClaimDetectionLabel.CLAIM

    # Invalid claim_type rejected
    with pytest.raises(ValidationError):
        TrainingClaim(
            claim_id="c_02",
            post_id="p_01",
            claim_text="Text",
            claim_type="INVALID_CLAIM_TYPE",
            provenance=prov,
        )

    # Check-worthiness out of range rejected
    with pytest.raises(ValidationError):
        TrainingClaim(
            claim_id="c_03",
            post_id="p_01",
            claim_text="Text",
            check_worthiness=1.5,
            provenance=prov,
        )


def test_atomic_claim_schema_compatibility():
    atomic = TrainingAtomicClaim(
        atomic_claim_id="at_01",
        parent_claim_id="c_01",
        original_text="Investor deposits ₹10,000 today",
        normalized_text="Investor deposits ₹10,000 today",
        claim_type="FINANCIAL",
        subject="Investor",
        predicate="deposits",
        value="10,000",
        currency="INR",
        temporal_context={"relative": "today"},
        polarity="POSITIVE",
        negated=False,
    )
    assert atomic.atomic_claim_id == "at_01"
    assert atomic.claim_type_enum == ClaimType.FINANCIAL
    assert atomic.currency == "INR"
    assert atomic.predicate == "deposits"


def test_query_schema_validation():
    prov = ProvenanceMetadata(
        source_type=ProvenanceSourceType.TRUSTLENS_ANNOTATED,
        source_name="Search Engine Integration",
    )
    query = TrainingQuery(
        query_id="q_01",
        claim_id="c_01",
        query_text="Crypto deposit guaranteed 50% monthly returns authorization",
        query_type=QueryType.FINANCIAL,
        language="en",
        provenance=prov,
    )
    assert query.query_type == QueryType.FINANCIAL

    # Invalid query_type rejected
    with pytest.raises(ValidationError):
        TrainingQuery(
            query_id="q_02",
            claim_id="c_01",
            query_text="Query",
            query_type="UNKNOWN_QUERY_TYPE",
            provenance=prov,
        )


def test_evidence_schema_and_relation_labels():
    prov = ProvenanceMetadata(
        source_type=ProvenanceSourceType.HUMAN_REVIEWED,
        source_name="Fact Checking Portal",
    )
    ev = TrainingEvidence(
        evidence_id="ev_01",
        claim_id="c_01",
        evidence_text="SEBI circular bans fixed return promises in securities trading.",
        source_url="https://sebi.gov.in/rules/circular_01.pdf",
        source_type="REGULATORY_CIRCULAR",
        retrieval_method="dense",
        relation_label=EvidenceRelationLabel.CONTRADICTS,
        provenance=prov,
    )
    assert ev.relation_label == EvidenceRelationLabel.CONTRADICTS

    # Invalid evidence relation label rejected
    with pytest.raises(ValidationError):
        TrainingEvidence(
            evidence_id="ev_02",
            claim_id="c_01",
            evidence_text="Text",
            source_type="WEB",
            retrieval_method="bm25",
            relation_label="PROBABLY_TRUE",
            provenance=prov,
        )


def test_risk_schema_validation():
    prov = ProvenanceMetadata(
        source_type=ProvenanceSourceType.TRUSTLENS_ANNOTATED,
        source_name="Risk Modeling Engine",
    )
    risk = TrainingRisk(
        risk_id="r_01",
        post_id="p_01",
        claim_ids=["c_01"],
        risk_level=RiskLevel.HIGH,
        risk_factors=["guaranteed_returns", "unregistered_entity"],
        provenance=prov,
    )
    assert risk.risk_level == RiskLevel.HIGH

    # Invalid risk level rejected
    with pytest.raises(ValidationError):
        TrainingRisk(
            risk_id="r_02",
            post_id="p_01",
            risk_level="CRITICAL_DANGER",
            provenance=prov,
        )


def test_annotation_metadata_and_quality_control():
    ann_high = AnnotationMetadata(
        annotator_id="annotator_42",
        confidence=AnnotationConfidence.HIGH,
        review_status=ReviewStatus.APPROVED,
    )
    assert ann_high.review_status == ReviewStatus.APPROVED

    # Low confidence must automatically flag for review
    ann_low = AnnotationMetadata(
        annotator_id="annotator_42",
        confidence=AnnotationConfidence.LOW,
        review_status=ReviewStatus.APPROVED,
    )
    assert ann_low.review_status == ReviewStatus.FLAGGED_FOR_REVIEW


def test_dataset_manifest_validation():
    manifest = DatasetManifest(
        dataset_id="trustlens_phase3_benchmark",
        dataset_version="1.0.0",
        source_summary="Curated multilingual social media verification benchmark",
        license="CC-BY-4.0",
        record_count=1000,
        language_distribution={"en": 500, "ta": 300, "hi": 200},
        label_distribution={"HIGH": 400, "MEDIUM": 300, "LOW": 200, "INSUFFICIENT": 100},
        splits_distribution={"train": 700, "validation": 150, "test": 150},
        leakage_group_counts={"campaigns": 120, "sources": 350},
    )
    assert manifest.record_count == 1000
    assert manifest.splits_distribution["train"] == 700


def test_split_and_leakage_grouping():
    split = SplitMetadata(
        split=SplitName.train,
        source_group_id="src_cluster_99",
        campaign_group_id="camp_fin_scam_01",
        post_family_id="fam_post_100",
        image_family_id="fam_img_200",
        translation_group_id="trans_pair_300",
    )
    assert split.split == SplitName.train
    assert split.campaign_group_id == "camp_fin_scam_01"


def test_synthetic_examples_validation():
    data = get_synthetic_examples()
    assert len(data["posts"]) == 10
    assert len(data["claims"]) == 11
    assert len(data["atomic_claims"]) == 5
    assert len(data["evidence"]) == 2
    assert len(data["risks"]) == 2

    # Verify all examples are flagged with is_example=True
    for post in data["posts"]:
        assert post.is_example is True
    for claim in data["claims"]:
        assert claim.is_example is True

    # Verify JSON round-trip
    json_path = Path("data/schema/examples.json")
    assert json_path.exists()
    content = json.loads(json_path.read_text(encoding="utf-8"))
    assert "posts" in content
    assert len(content["posts"]) == 10
