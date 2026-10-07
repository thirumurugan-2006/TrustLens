"""
TrustLens Phase 6B Retrieval Dataset & Hard-Negative Construction Test Suite.
Deterministic, offline tests verifying:
1. Retrieval schema validation
2. Query provenance and atomic claim linkage
3. Positive evidence selection (SUPPORTS/CONTRADICTS)
4. Hard-negative candidate selection
5. Negative safety (rejection of SUPPORTS/CONTRADICTS)
6. Duplicate pair detection
7. Multilingual retrieval coverage
8. Cross-language retrieval tracking
9. Translation leakage prevention
10. Campaign leakage prevention
11. Cluster-safe splitting
12. Test isolation
13. Dataset manifest generation
14. Dataset validator (valid and invalid cases)
15. Builder CLI dry-run
16. Validator CLI exit codes
17. Human-review metadata tracking
18. Retrieval evaluation metrics (Recall@K, MRR, nDCG@K)
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.claims.schemas import AtomicClaim
from app.input.schemas import AuthorInfo, Platform, PostContent, PostType
from app.retrieval_dataset.schemas import (
    NegativeEvidenceRecord,
    NegativeType,
    PositiveEvidenceRecord,
    RetrievalEvaluationItem,
    RetrievalExample,
    RetrievalPair,
)
from app.retrieval_dataset.query_builder import RetrievalQueryBuilder
from app.retrieval_dataset.positive_selector import PositiveEvidenceSelector
from app.retrieval_dataset.hard_negative_selector import HardNegativeSelector
from app.retrieval_dataset.cluster_manager import RetrievalClusterManager
from app.retrieval_dataset.splitter import RetrievalDatasetSplitter
from app.retrieval_dataset.pair_builder import RetrievalPairBuilder
from app.retrieval_dataset.manifest import RetrievalManifestBuilder
from app.retrieval_dataset.validator import RetrievalDatasetValidator
from app.retrieval_dataset.builder import RetrievalDatasetBuilder
from app.training.schemas import (
    ClaimType,
    EvidenceRelationLabel,
    ProvenanceMetadata,
    ProvenanceSourceType,
    QueryType,
    SplitMetadata,
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
)
from app.training.evaluation import EvaluationFramework


@pytest.fixture
def sample_training_claim():
    return TrainingClaim(
        claim_id="tl_claim_test_01",
        post_id="tl_post_test_01",
        claim_text="Guaranteed 35% weekly profit on crypto deposit via Telegram channel.",
        claim_type=ClaimType.FINANCIAL,
        language="en",
        provenance=ProvenanceMetadata(
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
            source_name="Regulatory Warnings",
        ),
        split_info=SplitMetadata(
            split=SplitName.train,
            campaign_group_id="camp_test_01",
            source_group_id="src_test_01",
        ),
    )


@pytest.fixture
def sample_training_evidence():
    return [
        TrainingEvidence(
            evidence_id="ev_pos_01",
            claim_id="tl_claim_test_01",
            evidence_text="Official SEBI Bulletin confirms that crypto solicitation is fraudulent.",
            relation_label=EvidenceRelationLabel.CONTRADICTS,
            source_type="REGULATORY_ALERT",
            provenance=ProvenanceMetadata(
                source_type=ProvenanceSourceType.PUBLIC_DATASET,
                source_name="SEBI Bulletins",
            ),
        ),
        TrainingEvidence(
            evidence_id="ev_neu_01",
            claim_id="tl_claim_other_99",
            evidence_text="General economic advisory notes that digital investment platforms require registration.",
            relation_label=EvidenceRelationLabel.NEUTRAL,
            source_type="FACT_CHECKING",
            provenance=ProvenanceMetadata(
                source_type=ProvenanceSourceType.PUBLIC_DATASET,
                source_name="Economic Advisory",
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Retrieval Schema Validation
# ---------------------------------------------------------------------------
def test_1_retrieval_schema_validation():
    """Verify RetrievalExample, PositiveEvidenceRecord, and NegativeEvidenceRecord schemas."""
    pos = PositiveEvidenceRecord(
        evidence_id="ev_pos_1",
        text="Official SEBI bulletin contradicts claim.",
        relation=EvidenceRelationLabel.CONTRADICTS,
        source_type=ProvenanceSourceType.PUBLIC_DATASET,
    )
    neg = NegativeEvidenceRecord(
        evidence_id="ev_neg_1",
        text="General stock exchange operating hours.",
        negative_type=NegativeType.HARD_TOPIC_NEGATIVE,
        source_type=ProvenanceSourceType.PUBLIC_DATASET,
    )
    example = RetrievalExample(
        retrieval_id="ret_001",
        post_id="post_001",
        claim_id="claim_001",
        atomic_claim_id="atomic_001",
        query_id="q_001",
        query="Guaranteed 20% return on deposit",
        positive=pos,
        negatives=[neg],
        query_language="en",
        evidence_language="en",
        cross_language=False,
        split=SplitName.train,
        cluster_id="cluster_001",
    )
    assert example.retrieval_id == "ret_001"
    assert example.positive.relation == EvidenceRelationLabel.CONTRADICTS
    assert example.negatives[0].negative_type == NegativeType.HARD_TOPIC_NEGATIVE


# ---------------------------------------------------------------------------
# 2. Query Linkage & Traceability
# ---------------------------------------------------------------------------
def test_2_query_linkage(sample_training_claim):
    """Verify that synthesized queries trace to post_id, claim_id, and atomic_claim_id."""
    builder = RetrievalQueryBuilder()
    atomic = [
        AtomicClaim(
            atomic_id="atomic_test_01",
            parent_claim_id=sample_training_claim.claim_id,
            original_text="Guaranteed 35% weekly profit",
            normalized_text="Guaranteed 35% weekly profit",
            subject="crypto deposit",
            predicate="guarantees profit",
            object="35% weekly",
        )
    ]
    queries = builder.build_queries_for_claim(sample_training_claim, atomic_claims=atomic)
    assert len(queries) >= 1
    q = queries[0]
    assert q["post_id"] == sample_training_claim.post_id
    assert q["claim_id"] == sample_training_claim.claim_id
    assert q["atomic_claim_id"].startswith("atomic_")
    atomic_q = [x for x in queries if x.get("variant_name") == "atomic_claim_variant"]
    if atomic_q:
        assert atomic_q[0]["atomic_claim_id"] == "atomic_test_01"
    assert len(q["query_text"]) > 0


# ---------------------------------------------------------------------------
# 3. Positive Evidence Selection
# ---------------------------------------------------------------------------
def test_3_positive_evidence_selection(sample_training_claim, sample_training_evidence):
    """Verify that PositiveEvidenceSelector accepts only SUPPORTS or CONTRADICTS."""
    selector = PositiveEvidenceSelector()
    pos = selector.select_positive(sample_training_claim.claim_id, sample_training_evidence)
    assert pos is not None
    assert pos.evidence_id == "ev_pos_01"
    assert pos.relation_label == EvidenceRelationLabel.CONTRADICTS

    # Test with SUPPORTS
    supports_ev = sample_training_evidence[0].model_copy(deep=True)
    supports_ev.relation_label = EvidenceRelationLabel.SUPPORTS
    assert selector.is_valid_positive(supports_ev) is True

    # Test with NEUTRAL / INSUFFICIENT
    neutral_ev = sample_training_evidence[1].model_copy(deep=True)
    assert selector.is_valid_positive(neutral_ev) is False


# ---------------------------------------------------------------------------
# 4. Hard-Negative Candidate Selection
# ---------------------------------------------------------------------------
def test_4_hard_negative_validation(sample_training_claim):
    """Verify HardNegativeSelector returns semantically plausible negatives."""
    selector = HardNegativeSelector()
    candidates = [
        TrainingEvidence(
            evidence_id="ev_cand_01",
            claim_id="other_claim_1",
            evidence_text="General economic advisory discusses crypto currency taxation and risk disclosure.",
            relation_label=EvidenceRelationLabel.NEUTRAL,
            provenance=ProvenanceMetadata(
                source_type=ProvenanceSourceType.PUBLIC_DATASET,
                source_name="Financial Regulatory Advisory",
            ),
        ),
        TrainingEvidence(
            evidence_id="ev_cand_02",
            claim_id="other_claim_2",
            evidence_text="Public company registry provides verification guidelines for registered brokers.",
            relation_label=EvidenceRelationLabel.INSUFFICIENT,
            provenance=ProvenanceMetadata(
                source_type=ProvenanceSourceType.PUBLIC_DATASET,
                source_name="Corporate Advisory",
            ),
        ),
    ]
    query_text = "crypto deposit weekly profit Telegram"
    negatives = selector.select_hard_negatives_for_claim(
        target_claim=sample_training_claim,
        query_text=query_text,
        candidate_pool=candidates,
        positive_evidence_id="ev_pos_01",
        max_negatives=2,
    )
    assert len(negatives) == 2
    assert all(isinstance(n, NegativeEvidenceRecord) for n in negatives)
    assert all(n.negative_type in NegativeType for n in negatives)


# ---------------------------------------------------------------------------
# 5. Negative Relation Safety Validation
# ---------------------------------------------------------------------------
def test_5_negative_relation_validation(sample_training_claim):
    """CRITICAL RULE: SUPPORTS and CONTRADICTS evidence must NEVER be accepted as negatives."""
    selector = HardNegativeSelector()
    invalid_candidates = [
        TrainingEvidence(
            evidence_id="ev_bad_sup",
            claim_id="other_claim_1",
            evidence_text="Official confirmation confirms scheme legitimacy.",
            relation_label=EvidenceRelationLabel.SUPPORTS,
            provenance=ProvenanceMetadata(
                source_type=ProvenanceSourceType.PUBLIC_DATASET,
                source_name="Regulatory Report",
            ),
        ),
        TrainingEvidence(
            evidence_id="ev_bad_con",
            claim_id="other_claim_2",
            evidence_text="Regulatory alert warns of illicit scam scheme.",
            relation_label=EvidenceRelationLabel.CONTRADICTS,
            provenance=ProvenanceMetadata(
                source_type=ProvenanceSourceType.PUBLIC_DATASET,
                source_name="Regulatory Report",
            ),
        ),
    ]
    negatives = selector.select_hard_negatives_for_claim(
        target_claim=sample_training_claim,
        query_text="Crypto deposit",
        candidate_pool=invalid_candidates,
        positive_evidence_id="ev_pos_01",
    )
    assert len(negatives) == 0  # Both invalid candidates rejected!

    # Schema-level rejection
    with pytest.raises(ValidationError):
        NegativeEvidenceRecord(
            evidence_id="ev_err",
            text="Fraud warning.",
            evidence_relation=EvidenceRelationLabel.CONTRADICTS,
            negative_type=NegativeType.HARD_TOPIC_NEGATIVE,
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
        )


# ---------------------------------------------------------------------------
# 6. Duplicate Detection
# ---------------------------------------------------------------------------
def test_6_duplicate_detection(tmp_path):
    """Verify validator flags duplicate (query, document) pairs."""
    data_dir = tmp_path / "retrieval"
    data_dir.mkdir()

    pos = PositiveEvidenceRecord(
        evidence_id="ev_pos_1",
        text="Official bulletin.",
        relation=EvidenceRelationLabel.CONTRADICTS,
        source_type=ProvenanceSourceType.PUBLIC_DATASET,
    )
    neg = NegativeEvidenceRecord(
        evidence_id="ev_neg_1",
        text="General advisory.",
        negative_type=NegativeType.HARD_TOPIC_NEGATIVE,
        source_type=ProvenanceSourceType.PUBLIC_DATASET,
    )
    ex1 = RetrievalExample(
        retrieval_id="ret_01",
        post_id="p1",
        claim_id="c1",
        atomic_claim_id="a1",
        query_id="q1",
        query="High return investment",
        positive=pos,
        negatives=[neg],
        query_language="en",
        evidence_language="en",
        split=SplitName.train,
        cluster_id="clust_01",
    )
    ex2 = ex1.model_copy(update={"retrieval_id": "ret_02"})

    with (data_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        f.write(ex1.model_dump_json() + "\n")
        f.write(ex2.model_dump_json() + "\n")

    validator = RetrievalDatasetValidator(str(data_dir))
    report = validator.validate()
    assert report["duplicate_pairs"] > 0
    assert report["is_valid"] is False


# ---------------------------------------------------------------------------
# 7. Multilingual Retrieval
# ---------------------------------------------------------------------------
def test_7_multilingual_retrieval(sample_training_claim):
    """Verify multilingual query generation (Tamil, Hindi, English)."""
    claim_ta = sample_training_claim.model_copy(deep=True)
    claim_ta.language = "ta"
    claim_ta.claim_text = "மாதம் 25% உத்தரவாத வருமானம் கிடைக்கும்."

    builder = RetrievalQueryBuilder()
    queries = builder.build_queries_for_claim(claim_ta)
    assert len(queries) >= 1
    assert queries[0]["language"] == "ta"


# ---------------------------------------------------------------------------
# 8. Cross-Language Retrieval Tracking
# ---------------------------------------------------------------------------
def test_8_cross_language_retrieval():
    """Verify cross-language tracking (e.g. Tamil query with English evidence)."""
    pos = PositiveEvidenceRecord(
        evidence_id="ev_en_01",
        text="SEBI issues investor caution notice regarding unregistered Telegram operations.",
        relation=EvidenceRelationLabel.CONTRADICTS,
        source_type=ProvenanceSourceType.PUBLIC_DATASET,
    )
    ex = RetrievalExample(
        retrieval_id="ret_xl_01",
        post_id="p_ta_01",
        claim_id="c_ta_01",
        atomic_claim_id="a_ta_01",
        query_id="q_ta_01",
        query="மாதம் 25% உத்தரவாத வருமானம்",
        positive=pos,
        negatives=[],
        query_language="ta",
        evidence_language="en",
        cross_language=True,
        split=SplitName.train,
        cluster_id="clust_01",
    )
    assert ex.cross_language is True
    assert ex.query_language != ex.evidence_language


# ---------------------------------------------------------------------------
# 9. Translation Leakage Prevention
# ---------------------------------------------------------------------------
def test_9_translation_leakage():
    """Verify translation variants belong to the same cluster."""
    p1 = TrainingPost(
        post_id="p_en",
        platform=Platform.telegram,
        author=AuthorInfo(username="user1"),
        post_type=PostType.text,
        content=PostContent(text="English post text"),
        provenance=ProvenanceMetadata(
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
            source_name="Test Source",
        ),
        split_info=SplitMetadata(
            split=SplitName.train,
            translation_group_id="trans_group_99",
        ),
    )
    p2 = TrainingPost(
        post_id="p_ta",
        platform=Platform.telegram,
        author=AuthorInfo(username="user1"),
        post_type=PostType.text,
        content=PostContent(text="Tamil post text"),
        provenance=ProvenanceMetadata(
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
            source_name="Test Source",
        ),
        split_info=SplitMetadata(
            split=SplitName.train,
            translation_group_id="trans_group_99",
        ),
    )
    cluster_mgr = RetrievalClusterManager()
    c1 = cluster_mgr.get_cluster_id(p1)
    c2 = cluster_mgr.get_cluster_id(p2)
    assert c1 == c2 == "trans_trans_group_99"


# ---------------------------------------------------------------------------
# 10. Campaign Leakage Prevention
# ---------------------------------------------------------------------------
def test_10_campaign_leakage():
    """Verify campaign posts belong to the same cluster."""
    p1 = TrainingPost(
        post_id="p_camp_a",
        platform=Platform.telegram,
        author=AuthorInfo(username="user1"),
        post_type=PostType.text,
        content=PostContent(text="Campaign variant A"),
        provenance=ProvenanceMetadata(
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
            source_name="Test Source",
        ),
        split_info=SplitMetadata(
            split=SplitName.train,
            campaign_group_id="camp_99",
        ),
    )
    p2 = TrainingPost(
        post_id="p_camp_b",
        platform=Platform.telegram,
        author=AuthorInfo(username="user1"),
        post_type=PostType.text,
        content=PostContent(text="Campaign variant B"),
        provenance=ProvenanceMetadata(
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
            source_name="Test Source",
        ),
        split_info=SplitMetadata(
            split=SplitName.train,
            campaign_group_id="camp_99",
        ),
    )
    cluster_mgr = RetrievalClusterManager()
    assert cluster_mgr.get_cluster_id(p1) == cluster_mgr.get_cluster_id(p2) == "camp_camp_99"


# ---------------------------------------------------------------------------
# 11. Cluster-Safe Split
# ---------------------------------------------------------------------------
def test_11_cluster_safe_split():
    """Verify zero overlap between splits when partitioned by cluster."""
    clusters = {f"clust_{i:02d}": [f"item_{i}"] for i in range(20)}
    splitter = RetrievalDatasetSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split_clusters(clusters)

    train_set = set(splits["train"])
    val_set = set(splits["validation"])
    test_set = set(splits["test"])

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    assert len(train_set) + len(val_set) + len(test_set) == len(clusters)


# ---------------------------------------------------------------------------
# 12. Test Isolation
# ---------------------------------------------------------------------------
def test_12_test_isolation():
    """Verify test set has 0 leakage and strictly positive count."""
    validator = RetrievalDatasetValidator("data/processed/retrieval")
    report = validator.validate()
    assert report["cross_split_leakage"] == 0
    assert report["test_count"] > 0
    assert report["is_valid"] is True


# ---------------------------------------------------------------------------
# 13. Manifest Generation
# ---------------------------------------------------------------------------
def test_13_manifest_generation():
    """Verify RetrievalManifestBuilder constructs valid manifest from dataset."""
    validator = RetrievalDatasetValidator("data/processed/retrieval")
    examples = validator.load_examples()
    manifest_builder = RetrievalManifestBuilder()
    manifest = manifest_builder.build_manifest(examples)

    assert manifest.dataset_version == "v0.1.0"
    assert manifest.source_dataset_version == "v0.2.0"
    assert manifest.record_count == len(examples)
    assert manifest.train_count > 0
    assert manifest.validation_count > 0
    assert manifest.test_count > 0
    assert manifest.human_reviewed_count >= int(0.20 * len(examples))
    assert manifest.duplicate_count == 0
    assert manifest.invalid_count == 0


# ---------------------------------------------------------------------------
# 14. Dataset Validator (Valid & Corrupted Cases)
# ---------------------------------------------------------------------------
def test_14_validator_pass_and_fail(tmp_path):
    """Verify validator accepts valid files and rejects corrupted files."""
    valid_validator = RetrievalDatasetValidator("data/processed/retrieval")
    res = valid_validator.validate()
    assert res["is_valid"] is True

    # Corrupted case
    corrupt_dir = tmp_path / "corrupt_retrieval"
    corrupt_dir.mkdir()
    pos = PositiveEvidenceRecord(
        evidence_id="ev_pos_1",
        text="Bulletin.",
        relation=EvidenceRelationLabel.CONTRADICTS,
        source_type=ProvenanceSourceType.PUBLIC_DATASET,
    )
    corrupted_data = {
        "retrieval_id": "ret_bad",
        "post_id": "p1",
        "claim_id": "c1",
        "atomic_claim_id": "a1",
        "query_id": "q1",
        "query": "Guaranteed money",
        "positive": pos.model_dump(),
        "negatives": [
            {
                "evidence_id": "ev_bad",
                "text": "Regulatory warning",
                "evidence_relation": "CONTRADICTS",
                "negative_type": "HARD_TOPIC_NEGATIVE",
                "source_type": "PUBLIC_DATASET",
            }
        ],
        "query_language": "en",
        "evidence_language": "en",
        "cross_language": False,
        "split": "train",
        "cluster_id": "c1",
    }
    with (corrupt_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(corrupted_data) + "\n")

    corrupt_validator = RetrievalDatasetValidator(str(corrupt_dir))
    res_corrupt = corrupt_validator.validate()
    assert res_corrupt["is_valid"] is False
    assert len(res_corrupt["errors"]) > 0


# ---------------------------------------------------------------------------
# 15. CLI Dry Run
# ---------------------------------------------------------------------------
def test_15_cli_dry_run():
    """Verify CLI dry run executes without modifying directory."""
    cmd = [
        sys.executable,
        "-m",
        "app.retrieval_dataset.builder",
        "--data-dir",
        "data/trustlens",
        "--output-dir",
        "data/processed/retrieval_dryrun_test",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "DRY RUN (No files written)" in proc.stdout
    assert not Path("data/processed/retrieval_dryrun_test").exists()


# ---------------------------------------------------------------------------
# 16. CLI Exit Codes
# ---------------------------------------------------------------------------
def test_16_cli_exit_codes():
    """Verify validator CLI returns exit code 0 for valid data."""
    cmd = [
        sys.executable,
        "-m",
        "app.retrieval_dataset.validator",
        "--data-dir",
        "data/processed/retrieval",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Overall:              VALID" in proc.stdout


# ---------------------------------------------------------------------------
# 17. Human Review Metadata Tracking
# ---------------------------------------------------------------------------
def test_17_human_review_metadata():
    """Verify human review metadata is tracked on >= 20% of examples."""
    validator = RetrievalDatasetValidator("data/processed/retrieval")
    examples = validator.load_examples()
    reviewed_examples = [
        ex
        for ex in examples
        if ex.human_reviewed
    ]
    assert len(reviewed_examples) >= int(0.20 * len(examples))
    sample = reviewed_examples[0]
    assert sample.reviewer_id.startswith("retrieval_reviewer_") or sample.reviewer_id.startswith("annotator_")
    assert sample.review_confidence is not None


# ---------------------------------------------------------------------------
# 18. Retrieval Evaluation Metrics
# ---------------------------------------------------------------------------
def test_18_retrieval_evaluation_metrics():
    """Verify EvaluationFramework calculation of Recall@K, Precision@K, MRR, and nDCG@K."""
    ranked = [
        ["ev_2", "ev_1", "ev_3", "ev_4", "ev_5"],
        ["ev_10", "ev_20", "ev_30", "ev_40", "ev_50"],
        ["ev_100", "ev_200", "ev_300", "ev_400", "ev_500"],
    ]
    relevant = [
        ["ev_1"],
        ["ev_99"],
        ["ev_100"],
    ]

    metrics = EvaluationFramework.calculate_retrieval_metrics(
        ranked_evidence_ids=ranked,
        ground_truth_evidence_ids=relevant,
        k_values=[1, 5, 10, 20],
    )

    assert pytest.approx(metrics["recall_at_1"], 0.01) == 0.3333
    assert pytest.approx(metrics["recall_at_5"], 0.01) == 0.6667
    assert "recall_at_10" in metrics
    assert "recall_at_20" in metrics
    assert "precision_at_1" in metrics
    assert "precision_at_5" in metrics
    assert pytest.approx(metrics["mrr"], 0.01) == 0.50
    assert metrics["ndcg_at_5"] > 0.0

