import csv
import json
from pathlib import Path

import pytest

from app.claims.schemas import Claim, SourceSpan
from app.dataset.annotation_queue import AnnotationQueue, AnnotationQueueError
from app.dataset.clusterer import DatasetClusterer
from app.dataset.collector import DatasetCollector
from app.dataset.deduplicator import (
    DatasetDeduplicator,
    compute_simulated_phash,
    compute_text_hash,
    hamming_distance,
)
from app.dataset.manifest import DatasetManifestBuilder, Phase4DatasetManifest
from app.dataset.normalizer import DatasetNormalizer, NormalizationError
from app.dataset.provenance import ProvenanceError, ProvenanceManager
from app.dataset.quality_control import DatasetQualityControl
from app.dataset.schemas import (
    AnnotationTaskStatus,
    DuplicateType,
    HumanAnnotationSubmission,
    RawDataRecord,
    SemanticAutoSuggestions,
)
from app.dataset.splitter import DatasetSplitter
from app.dataset.validator import DatasetValidator
from app.input.schemas import Platform, PostContent, PostMedia, PostType
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
def sample_provenance():
    return ProvenanceManager.create_provenance(
        source_type=ProvenanceSourceType.PUBLIC_DATASET,
        source_name="SyntheticResearchSuite",
        source_id="item_001",
        license_str="CC-BY-4.0",
        collection_method="automated_test_fixture",
    )


@pytest.fixture
def base_training_post(sample_provenance):
    return TrainingPost(
        post_id="post_test_001",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Invest ₹5,000 today and receive guaranteed ₹15,000 within 2 hours!"),
        timestamp="2026-10-07T10:00:00Z",
        language_info=LanguageMetadata(
            primary="en",
            languages=["en"],
            script=["Latin"],
            original_text="Invest ₹5,000 today and receive guaranteed ₹15,000 within 2 hours!",
            normalized_text="Invest ₹5,000 today and receive guaranteed ₹15,000 within 2 hours!",
        ),
        provenance=sample_provenance,
        is_example=False,
    )


# ---------------------------------------------------------------------------
# 1. JSON Import
# ---------------------------------------------------------------------------

def test_json_import(tmp_path):
    normalizer = DatasetNormalizer()
    json_file = tmp_path / "data.json"
    data = [
        {"post_id": "p1", "text": "Earn daily returns now", "platform": "reddit"},
        {"post_id": "p2", "text": "Free gifts giveaway", "platform": "twitter"},
    ]
    json_file.write_text(json.dumps(data), encoding="utf-8")

    records = normalizer.parse_file(json_file)
    assert len(records) == 2
    assert records[0].raw_payload["post_id"] == "p1"
    assert records[1].raw_payload["post_id"] == "p2"


# ---------------------------------------------------------------------------
# 2. JSONL Import
# ---------------------------------------------------------------------------

def test_jsonl_import(tmp_path):
    normalizer = DatasetNormalizer()
    jsonl_file = tmp_path / "data.jsonl"
    lines = [
        json.dumps({"post_id": "jl1", "text": "Contact +919876543210 for job offers", "platform": "reddit"}),
        json.dumps({"post_id": "jl2", "text": "Send payment to test@okaxis", "platform": "reddit"}),
    ]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    records = normalizer.parse_file(jsonl_file)
    assert len(records) == 2
    assert records[0].raw_payload["post_id"] == "jl1"
    assert records[1].raw_payload["post_id"] == "jl2"


# ---------------------------------------------------------------------------
# 3. CSV Import
# ---------------------------------------------------------------------------

def test_csv_import(tmp_path):
    normalizer = DatasetNormalizer()
    csv_file = tmp_path / "data.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["post_id", "text", "platform"])
        writer.writeheader()
        writer.writerow({"post_id": "csv1", "text": "Instant loan approval without CIBIL", "platform": "reddit"})
        writer.writerow({"post_id": "csv2", "text": "Claim your subsidy reward", "platform": "reddit"})

    records = normalizer.parse_file(csv_file)
    assert len(records) == 2
    assert records[0].raw_payload["post_id"] == "csv1"
    assert records[1].raw_payload["post_id"] == "csv2"


# ---------------------------------------------------------------------------
# 4. TXT Import
# ---------------------------------------------------------------------------

def test_txt_import(tmp_path):
    normalizer = DatasetNormalizer()
    txt_file = tmp_path / "data.txt"
    content = "Paragraph 1: Double your bitcoin in 24 hours.\n\nParagraph 2: Click http://scam-site.org for login."
    txt_file.write_text(content, encoding="utf-8")

    records = normalizer.parse_file(txt_file)
    assert len(records) == 2
    assert "Paragraph 1" in records[0].raw_payload["text"]
    assert "Paragraph 2" in records[1].raw_payload["text"]


# ---------------------------------------------------------------------------
# 5. Provenance Preservation
# ---------------------------------------------------------------------------

def test_provenance_preservation():
    # Valid creation
    prov = ProvenanceManager.create_provenance(
        source_type="PUBLIC_DATASET",
        source_name="GovAdvisoryArchive",
        source_id="adv_2026_09",
        license_str="Open-Gov-License",
    )
    assert prov.source_type == ProvenanceSourceType.PUBLIC_DATASET
    assert prov.source_name == "GovAdvisoryArchive"
    assert prov.source_id == "adv_2026_09"

    # Validation succeeds
    ProvenanceManager.validate_provenance(prov)

    # Rejection of invalid source type
    with pytest.raises(ProvenanceError, match="Invalid source_type"):
        ProvenanceManager.create_provenance(
            source_type="NON_EXISTENT_TYPE",
            source_name="AnySource",
        )

    # Rejection of empty source name
    with pytest.raises(ProvenanceError, match="source_name is required"):
        ProvenanceManager.create_provenance(
            source_type=ProvenanceSourceType.USER_PROVIDED,
            source_name="",
        )


# ---------------------------------------------------------------------------
# 6. Normalization
# ---------------------------------------------------------------------------

def test_normalization(sample_provenance):
    normalizer = DatasetNormalizer()
    raw = RawDataRecord(
        format="json",
        raw_payload={
            "post_id": "norm_001",
            "text": "  Exclusive offer!   Invest ₹10,000 get 20% return.  ",
            "platform": "reddit",
        },
    )
    post = normalizer.normalize_record(raw, sample_provenance)

    assert post.post_id == "norm_001"
    assert post.content.text == "Exclusive offer! Invest ₹10,000 get 20% return."
    assert post.metadata["original_text"] == "  Exclusive offer!   Invest ₹10,000 get 20% return.  "
    assert post.language_info.primary == "en"
    assert post.provenance.source_name == "SyntheticResearchSuite"


# ---------------------------------------------------------------------------
# 7. L2 Validation Integration
# ---------------------------------------------------------------------------

def test_l2_validation_integration(base_training_post):
    # Valid post passes
    res_valid = validate_post(base_training_post)
    assert res_valid.valid is True

    # Missing timestamp or invalid platform triggers L2 validation rejection
    invalid_post = base_training_post.model_copy(deep=True)
    invalid_post.platform = Platform.unknown
    res_invalid = validate_post(invalid_post)
    assert res_invalid.valid is False


# ---------------------------------------------------------------------------
# 8. Duplicate Detection (Exact & Near-Duplicate)
# ---------------------------------------------------------------------------

def test_duplicate_detection(base_training_post, sample_provenance):
    dedup = DatasetDeduplicator(text_similarity_threshold=0.85)

    # First registration is canonical
    d1 = dedup.check_and_register(base_training_post)
    assert not d1.is_duplicate

    # Exact duplicate
    exact_copy = base_training_post.model_copy(deep=True)
    exact_copy.post_id = "post_test_002"
    d2 = dedup.check_and_register(exact_copy)
    assert d2.is_duplicate
    assert d2.duplicate_type == DuplicateType.EXACT
    assert d2.duplicate_of == base_training_post.post_id

    # Near duplicate with minor variation
    near_copy = base_training_post.model_copy(deep=True)
    near_copy.post_id = "post_test_003"
    near_copy.content.text = "Invest ₹5,000 today and receive guaranteed ₹15,000 in 2 hours!"
    d3 = dedup.check_and_register(near_copy)
    assert d3.is_duplicate
    assert d3.duplicate_type in (DuplicateType.TEXT_SIMILARITY, DuplicateType.EXACT)


# ---------------------------------------------------------------------------
# 9. pHash Grouping Interface
# ---------------------------------------------------------------------------

def test_phash_grouping_interface(sample_provenance):
    dedup = DatasetDeduplicator(image_hamming_threshold=5)

    p1 = TrainingPost(
        post_id="img_post_1",
        platform=Platform.reddit,
        post_type=PostType.image_text,
        content=PostContent(text="Screenshot A"),
        media=PostMedia(images=["screenshot_proof_01.png"]),
        timestamp="2026-10-07T10:00:00Z",
        provenance=sample_provenance,
    )
    d1 = dedup.check_and_register(p1)
    assert not d1.is_duplicate

    # Same image path produces 0 Hamming distance
    p2 = TrainingPost(
        post_id="img_post_2",
        platform=Platform.reddit,
        post_type=PostType.image_text,
        content=PostContent(text="Screenshot B with distinct caption"),
        media=PostMedia(images=["screenshot_proof_01.png"]),
        timestamp="2026-10-07T10:00:00Z",
        provenance=sample_provenance,
    )
    d2 = dedup.check_and_register(p2)
    assert d2.is_duplicate
    assert d2.duplicate_type == DuplicateType.IMAGE_PHASH
    assert d2.duplicate_of == "img_post_1"


# ---------------------------------------------------------------------------
# 10. Translation Cluster Handling
# ---------------------------------------------------------------------------

def test_translation_cluster_handling(sample_provenance):
    clusterer = DatasetClusterer()

    p_en = TrainingPost(
        post_id="post_en",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Send money to get rich fast"),
        timestamp="2026-10-07T10:00:00Z",
        provenance=sample_provenance,
    )
    p_ta = TrainingPost(
        post_id="post_ta",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="பணம் அனுப்பி சீக்கிரம் பணக்காரர் ஆகுங்கள்"),
        timestamp="2026-10-07T10:00:00Z",
        provenance=sample_provenance,
    )

    clusterer.register_post(p_en, translation_group_id="trans_scam_01")
    clusterer.register_post(p_ta, translation_group_id="trans_scam_01")

    assert clusterer.get_composite_cluster("post_en") == clusterer.get_composite_cluster("post_ta")


# ---------------------------------------------------------------------------
# 11. Campaign Cluster Handling
# ---------------------------------------------------------------------------

def test_campaign_cluster_handling(sample_provenance):
    clusterer = DatasetClusterer()

    p_camp1 = TrainingPost(
        post_id="post_c1",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Contact support immediately via payment handle support@okhdfcbank"),
        timestamp="2026-10-07T10:00:00Z",
        provenance=sample_provenance,
    )
    p_camp2 = TrainingPost(
        post_id="post_c2",
        platform=Platform.reddit,
        post_type=PostType.text,
        content=PostContent(text="Urgent recharge payment needed at support@okhdfcbank"),
        timestamp="2026-10-07T10:00:00Z",
        provenance=sample_provenance,
    )

    c1 = clusterer.register_post(p_camp1)
    c2 = clusterer.register_post(p_camp2)

    assert c1.campaign_group_id == "camp_upi:support@okhdfcbank"
    assert clusterer.get_composite_cluster("post_c1") == clusterer.get_composite_cluster("post_c2")


# ---------------------------------------------------------------------------
# 12. Annotation Queue Lifecycle
# ---------------------------------------------------------------------------

def test_annotation_queue_lifecycle(base_training_post):
    queue = AnnotationQueue()

    # Enqueue
    task = queue.enqueue(base_training_post, priority=2)
    assert task.status == AnnotationTaskStatus.PENDING

    # Assign
    assigned_task = queue.assign(task.annotation_task_id, annotator_id="annotator_expert_1")
    assert assigned_task.status == AnnotationTaskStatus.IN_PROGRESS
    assert assigned_task.assigned_annotator == "annotator_expert_1"

    # Submit
    sub = HumanAnnotationSubmission(
        annotator_id="annotator_expert_1",
        claim_detection=ClaimDetectionLabel.CLAIM,
        claim_type=ClaimType.FINANCIAL,
        risk_level=RiskLevel.HIGH,
        confidence=AnnotationConfidence.HIGH,
    )
    submitted_task = queue.submit_annotation(task.annotation_task_id, sub)
    assert submitted_task.status == AnnotationTaskStatus.SUBMITTED

    # Accept
    accepted_task = queue.accept(task.annotation_task_id)
    assert accepted_task.status == AnnotationTaskStatus.ACCEPTED
    assert accepted_task.consensus_submission is not None
    assert accepted_task.consensus_submission.risk_level == RiskLevel.HIGH


# ---------------------------------------------------------------------------
# 13. Label Validation & QC
# ---------------------------------------------------------------------------

def test_label_validation(base_training_post, sample_provenance):
    # Post QC
    qc_post = DatasetQualityControl.validate_post(base_training_post)
    assert qc_post.passed is True

    # Claim QC
    claim = TrainingClaim(
        post_id=base_training_post.post_id,
        claim_text="Guaranteed ₹15,000 return",
        claim_type=ClaimType.FINANCIAL,
        detection_label=ClaimDetectionLabel.CLAIM,
        check_worthiness=1.0,
        provenance=sample_provenance,
    )
    qc_claim = DatasetQualityControl.validate_claim(claim, existing_post_ids={base_training_post.post_id})
    assert qc_claim.passed is True

    # Evidence QC
    evidence = TrainingEvidence(
        claim_id=claim.claim_id,
        evidence_text="Official SEBI registry shows no authorization for entity.",
        relation_label=EvidenceRelationLabel.CONTRADICTS,
        provenance=sample_provenance,
    )
    qc_evidence = DatasetQualityControl.validate_evidence(evidence, existing_claim_ids={claim.claim_id})
    assert qc_evidence.passed is True

    # Broken Reference Claim Check
    broken_claim = claim.model_copy(deep=True)
    broken_claim.post_id = "non_existent_post"
    qc_broken = DatasetQualityControl.validate_claim(broken_claim, existing_post_ids={base_training_post.post_id})
    assert qc_broken.passed is False
    assert any("BROKEN_REFERENCES" in r for r in qc_broken.reason_codes)


# ---------------------------------------------------------------------------
# 14. Leakage-Safe Split
# ---------------------------------------------------------------------------

def test_leakage_safe_split(sample_provenance):
    splitter = DatasetSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    clusterer = DatasetClusterer()

    posts = []
    for i in range(10):
        p = TrainingPost(
            post_id=f"p_{i}",
            platform=Platform.reddit,
            post_type=PostType.text,
            content=PostContent(text=f"Campaign text variant {i}"),
            timestamp="2026-10-07T10:00:00Z",
            provenance=sample_provenance,
        )
        posts.append(p)
        # Put first 3 posts in the same campaign cluster
        if i < 3:
            clusterer.register_post(p, campaign_group_id="camp_target_shared")
        else:
            clusterer.register_post(p, campaign_group_id=f"camp_individual_{i}")

    split_map = splitter.split_posts(posts, clusterer)

    # First 3 posts must belong to the exact same split
    assert split_map["p_0"] == split_map["p_1"] == split_map["p_2"]


# ---------------------------------------------------------------------------
# 15. Synthetic Exclusion from Test
# ---------------------------------------------------------------------------

def test_synthetic_exclusion_from_test():
    splitter = DatasetSplitter()
    clusterer = DatasetClusterer()

    prov_synth = ProvenanceManager.create_provenance(
        source_type=ProvenanceSourceType.SYNTHETIC,
        source_name="SyntheticGenerator",
    )
    prov_real = ProvenanceManager.create_provenance(
        source_type=ProvenanceSourceType.LOCAL_DATA,
        source_name="RealInvestigationData",
    )

    posts = []
    # 5 synthetic posts
    for i in range(5):
        p = TrainingPost(
            post_id=f"synth_{i}",
            platform=Platform.reddit,
            post_type=PostType.text,
            content=PostContent(text=f"Synthetic post {i}"),
            timestamp="2026-10-07T10:00:00Z",
            provenance=prov_synth,
            is_example=True,
        )
        clusterer.register_post(p)
        posts.append(p)

    # 15 real posts
    for i in range(15):
        p = TrainingPost(
            post_id=f"real_{i}",
            platform=Platform.reddit,
            post_type=PostType.text,
            content=PostContent(text=f"Real observed post {i}"),
            timestamp="2026-10-07T10:00:00Z",
            provenance=prov_real,
            is_example=False,
        )
        clusterer.register_post(p)
        posts.append(p)

    split_map = splitter.split_posts(posts, clusterer)

    for i in range(5):
        assert split_map[f"synth_{i}"] != SplitName.test, "Synthetic record found in test split!"


# ---------------------------------------------------------------------------
# 16. Manifest Generation
# ---------------------------------------------------------------------------

def test_manifest_generation(base_training_post, tmp_path):
    post_train = base_training_post.model_copy(deep=True)
    post_train.split_info = SplitMetadata(split=SplitName.train)

    manifest = DatasetManifestBuilder.build_manifest(
        posts=[post_train],
        duplicate_count=0,
        rejected_count=0,
        cluster_count=1,
    )

    assert isinstance(manifest, Phase4DatasetManifest)
    assert manifest.record_count == 1
    assert manifest.train_count == 1
    assert manifest.language_distribution.get("en") == 1

    out_file = tmp_path / "dataset_manifest.json"
    DatasetManifestBuilder.save_manifest(manifest, out_file)
    assert out_file.is_file()


# ---------------------------------------------------------------------------
# 17. Dataset Integrity Validation
# ---------------------------------------------------------------------------

def test_dataset_integrity_validation(base_training_post):
    validator = DatasetValidator()

    # Valid scenario
    valid_post = base_training_post.model_copy(deep=True)
    valid_post.split_info = SplitMetadata(split=SplitName.train)
    report = validator.validate_dataset([valid_post])
    assert report.valid is True
    assert report.total_records == 1

    # Synthetic leakage violation
    leaked_post = base_training_post.model_copy(deep=True)
    leaked_post.is_example = True
    leaked_post.split_info = SplitMetadata(split=SplitName.test)
    report_bad = validator.validate_dataset([leaked_post])
    assert report_bad.valid is False
    assert report_bad.synthetic_in_test_violations == 1


# ---------------------------------------------------------------------------
# 18. Dry-Run Mode
# ---------------------------------------------------------------------------

def test_dry_run_mode(tmp_path):
    collector = DatasetCollector(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
        rejected_dir=str(tmp_path / "rejected"),
        queue_dir=str(tmp_path / "queue"),
    )

    input_file = tmp_path / "incoming.json"
    data = [
        {"post_id": "dry_1", "text": "Instant profit scheme", "platform": "reddit"},
        {"post_id": "dry_2", "text": "Instant profit scheme", "platform": "reddit"},
    ]
    input_file.write_text(json.dumps(data), encoding="utf-8")

    summary = collector.process_file(
        input_path=str(input_file),
        source_type="PUBLIC_DATASET",
        source_name="DryRunSource",
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["records_discovered"] == 2
    assert summary["records_valid"] == 2
    assert summary["duplicates"] == 1

    # Ensure no physical files were written in dry-run mode
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "processed").exists()
