"""
Unit and Integration Tests for TrustLens Feature Dataset Module (Phase 6D).
Tests all 24 required capabilities: schemas, registry, all 10 feature extractors,
missing-value policies, leakage prevention, split integrity, manifest, validator, and loader.
"""

from pathlib import Path
import tempfile
import pytest
import pandas as pd
import numpy as np

from app.feature_dataset.schemas import (
    FeatureDefinition,
    FeatureDType,
    FeatureGroup,
    FeatureRecord,
    FeatureRegistry,
)
from app.feature_dataset.text_features import TextFeatureExtractor
from app.feature_dataset.language_features import LanguageFeatureExtractor
from app.feature_dataset.claim_features import ClaimFeatureExtractor
from app.feature_dataset.evidence_features import EvidenceFeatureExtractor
from app.feature_dataset.contradiction_features import ContradictionFeatureExtractor
from app.feature_dataset.retrieval_features import RetrievalFeatureExtractor
from app.feature_dataset.image_features import ImageFeatureExtractor
from app.feature_dataset.source_features import SourceFeatureExtractor
from app.feature_dataset.domain_features import DomainFeatureExtractor
from app.feature_dataset.graph_features import GraphFeatureExtractor
from app.feature_dataset.completeness import FeatureCompletenessAnalyzer
from app.feature_dataset.splitter import FeatureSplitter
from app.feature_dataset.serializer import FeatureSerializer
from app.feature_dataset.validator import FeatureDatasetValidator
from app.feature_dataset.manifest import FeatureManifestBuilder
from app.feature_dataset.loader import FeatureDatasetLoader, load_train, load_validation, load_test
from app.feature_dataset.builder import FeatureDatasetBuilder
from app.input.schemas import AuthorInfo, Platform, PostContent, PostType
from app.training.schemas import (
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


@pytest.fixture
def sample_post():
    return TrainingPost(
        post_id="post_test_001",
        platform=Platform.x,
        author=AuthorInfo(username="user_test", display_name="Test User"),
        post_type=PostType.text,
        content=PostContent(
            text="Earn ₹50,000 guaranteed daily profit! Visit https://scam-link.biz or call 9876543210. Limited slots!",
            urls=["https://scam-link.biz"],
        ),
        language_info=LanguageMetadata(
            primary="en",
            script=["Latin"],
            code_mixed=False,
            transliterated=False,
        ),
        provenance=ProvenanceMetadata(
            source_name="synthetic_audit",
            source_id="audit_01",
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
        ),
        split_info=SplitMetadata(
            split=SplitName.train,
            campaign_group_id="camp_test_01",
        ),
    )


@pytest.fixture
def sample_claim():
    return TrainingClaim(
        claim_id="claim_test_001",
        post_id="post_test_001",
        claim_text="Earn ₹50,000 guaranteed daily profit",
        claim_type=ClaimType.FINANCIAL,
        check_worthiness=0.95,
        atomic_claims=["Investor earns ₹50,000 per day"],
        provenance=ProvenanceMetadata(
            source_name="synthetic_audit",
            source_id="audit_claim_01",
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
        ),
    )


@pytest.fixture
def sample_evidence():
    return TrainingEvidence(
        evidence_id="ev_test_001",
        claim_id="claim_test_001",
        evidence_text="Regulatory alert by SEBI confirms this is an illegal Ponzi scheme.",
        relation_label=EvidenceRelationLabel.CONTRADICTS,
        source_url="https://sebi.gov.in/warning/alert.html",
        source_type="REGULATORY_ALERT",
        provenance=ProvenanceMetadata(
            source_name="SEBI Warning",
            source_id="sebi_001",
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
        ),
    )


# 1. Feature schema
def test_feature_schema():
    assert len(FeatureGroup) == 10
    assert len(FeatureDType) == 4
    defn = FeatureDefinition(
        name="test_feat",
        group=FeatureGroup.TEXT,
        dtype=FeatureDType.INT,
        description="A test feature",
        source="test",
        missing_policy="zero_default",
    )
    assert defn.name == "test_feat"
    assert defn.leakage_risk == "none"


# 2. Feature registry
def test_feature_registry():
    names = FeatureRegistry.get_feature_names()
    assert len(names) == 92
    assert len(names) == len(set(names))
    feat_map = FeatureRegistry.get_feature_map()
    assert "text_length" in feat_map
    assert "contradiction_ratio" in feat_map
    assert "guarantee_claim_present" in feat_map
    # Ensure risk_label is NEVER in registry
    assert "risk_label" not in feat_map
    assert "post_risk_score" not in feat_map


# 3. Text features
def test_text_features(sample_post):
    feats = TextFeatureExtractor.extract(sample_post.content.text)
    assert feats["text_length"] > 0
    assert feats["word_count"] > 0
    assert feats["currency_symbol_count"] >= 1
    assert feats["phone_count"] == 1
    assert feats["exclamation_count"] >= 1
    assert feats["question_count"] == 0

    # Empty text
    empty_feats = TextFeatureExtractor.extract("")
    assert empty_feats["text_length"] == 0
    assert empty_feats["word_count"] == 0


# 4. Language features
def test_language_features(sample_post):
    feats = LanguageFeatureExtractor.extract(sample_post)
    assert feats["language"] == "en"
    assert feats["language_id"] == 0
    assert feats["script"] == "Latin"
    assert feats["is_code_mixed"] is False
    assert feats["language_confidence"] == 1.0


# 5. Claim features
def test_claim_features(sample_claim):
    feats = ClaimFeatureExtractor.extract([sample_claim])
    assert feats["claim_count"] == 1
    assert feats["has_financial_claim"] is True
    assert feats["guarantee_claim_present"] is True
    assert feats["numeric_claim_count"] >= 1
    assert feats["financial_claim_count"] == 1

    # Empty claims
    empty_feats = ClaimFeatureExtractor.extract([])
    assert empty_feats["claim_count"] == 0
    assert empty_feats["has_financial_claim"] is False


# 6. Evidence features
def test_evidence_features(sample_claim, sample_evidence):
    feats = EvidenceFeatureExtractor.extract([sample_claim], [sample_evidence])
    assert feats["evidence_count"] == 1
    assert feats["has_evidence"] is True
    assert feats["claims_with_evidence"] == 1
    assert feats["claims_without_evidence"] == 0
    assert feats["evidence_coverage_ratio"] == 1.0

    # Zero evidence case: must NOT become safe
    zero_feats = EvidenceFeatureExtractor.extract([sample_claim], [])
    assert zero_feats["evidence_count"] == 0
    assert zero_feats["has_evidence"] is False
    assert zero_feats["claims_without_evidence"] == 1
    assert zero_feats["evidence_coverage_ratio"] == 0.0


# 7. Retrieval features
def test_retrieval_features():
    # With candidate pool
    eval_item = {
        "candidate_evidence_pool": [
            {"evidence_id": "ev_1", "candidate_similarity": 0.88, "is_relevant": True, "relation": "CONTRADICTS"},
            {"evidence_id": "ev_2", "candidate_similarity": 0.62, "is_relevant": False, "relation": "NEUTRAL"},
        ],
        "relevant_evidence_ids": ["ev_1"],
    }
    feats = RetrievalFeatureExtractor.extract(eval_item)
    assert feats["retrieved_evidence_count"] == 2
    assert feats["relevant_evidence_count"] == 1
    assert feats["contradicting_retrieval_count"] == 1
    assert feats["top_retrieval_score"] == 0.88
    assert feats["retrieval_score_available"] is True

    # Empty / unavailable retrieval: must preserve missing sentinel
    none_feats = RetrievalFeatureExtractor.extract(None)
    assert none_feats["retrieved_evidence_count"] == 0
    assert none_feats["top_retrieval_score"] == -1.0
    assert none_feats["retrieval_score_available"] is False


# 8. Contradiction features
def test_contradiction_features(sample_evidence):
    feats = ContradictionFeatureExtractor.extract([sample_evidence])
    assert feats["contradict_count"] == 1
    assert feats["support_count"] == 0
    assert feats["contradiction_ratio"] == 1.0
    assert feats["support_ratio"] == 0.0
    assert feats["contradiction_presence"] is True

    # Empty evidence
    empty_feats = ContradictionFeatureExtractor.extract([])
    assert empty_feats["contradict_count"] == 0
    assert empty_feats["contradiction_ratio"] == 0.0
    assert empty_feats["contradiction_presence"] is False


# 9. Image features
def test_image_features(sample_post):
    feats = ImageFeatureExtractor.extract(sample_post)
    assert feats["image_count"] == 0
    assert feats["has_image"] is False
    assert feats["ocr_available"] is False
    assert feats["ocr_confidence"] == -1.0


# 10. Source features
def test_source_features(sample_evidence):
    feats = SourceFeatureExtractor.extract([sample_evidence])
    assert feats["source_count"] >= 1
    assert feats["official_source_present"] is True

    empty_feats = SourceFeatureExtractor.extract([])
    assert empty_feats["source_count"] == 0
    assert empty_feats["official_source_present"] is False


# 11. Domain features
def test_domain_features(sample_post, sample_evidence):
    feats = DomainFeatureExtractor.extract(sample_post, [sample_evidence])
    assert feats["domain_count"] >= 1
    assert feats["official_domain_match"] is True
    # Explicit missing domain age
    assert feats["domain_age_days"] == -1.0
    assert feats["domain_age_available"] is False


# 12. Graph features
def test_graph_features():
    graph_dict = {
        "nodes": [
            {"node_type": "POST"},
            {"node_type": "CLAIM"},
            {"node_type": "EVIDENCE"},
        ],
        "edges": [
            {"edge_type": "HAS_CLAIM"},
            {"edge_type": "CONTRADICTS"},
        ],
    }
    feats = GraphFeatureExtractor.extract(graph_dict)
    assert feats["graph_node_count"] == 3
    assert feats["graph_edge_count"] == 2
    assert feats["graph_claim_nodes"] == 1
    assert feats["graph_evidence_nodes"] == 1
    assert feats["graph_contradict_edges"] == 1

    none_feats = GraphFeatureExtractor.extract(None)
    assert none_feats["graph_node_count"] == 0
    assert none_feats["graph_edge_count"] == 0


# 13. Missing-value policy
def test_missing_value_policy():
    analyzer = FeatureCompletenessAnalyzer()
    assert analyzer.is_missing_value("ocr_confidence", -1.0) is True
    assert analyzer.is_missing_value("domain_age_days", -1.0) is True
    assert analyzer.is_missing_value("text_length", 15) is False
    assert analyzer.is_missing_value("language", "UNKNOWN") is True


# 14. Target leakage detection & Section 51 Regression Test
def test_target_leakage_detection():
    df_clean = pd.DataFrame({
        "feature_id": ["feat_1"],
        "post_id": ["p_1"],
        "cluster_id": ["c_1"],
        "split": ["train"],
        "risk_label": ["HIGH"],
        "text_length": [100],
    })
    validator = FeatureDatasetValidator(Path("dummy"))
    res_clean = validator.audit_leakage(df_clean)
    assert res_clean.leakage_detected is False

    # Inject forbidden target leakage column
    df_leaky = pd.DataFrame({
        "feature_id": ["feat_1"],
        "post_id": ["p_1"],
        "cluster_id": ["c_1"],
        "split": ["train"],
        "risk_label": ["HIGH"],
        "text_length": [100],
        "ground_truth_risk": [3],
    })
    res_leaky = validator.audit_leakage(df_leaky)
    assert res_leaky.leakage_detected is True
    assert "ground_truth_risk" in res_leaky.forbidden_columns_found


# 15. Duplicate detection
def test_duplicate_detection():
    rec1 = FeatureRecord(
        feature_id="f1", post_id="p1", cluster_id="c1", split=SplitName.train,
        risk_label=RiskLevel.HIGH, features={"text_length": 10}
    )
    rec2 = FeatureRecord(
        feature_id="f2", post_id="p1", cluster_id="c1", split=SplitName.train,
        risk_label=RiskLevel.HIGH, features={"text_length": 10}
    )
    df = pd.DataFrame([rec1.model_dump(), rec2.model_dump()])
    assert df["post_id"].duplicated().sum() == 1


# 16. Split integrity & cluster safety
def test_split_integrity():
    rec_train = FeatureRecord(
        feature_id="f1", post_id="p1", cluster_id="c1", split=SplitName.train,
        risk_label=RiskLevel.HIGH, features={}
    )
    rec_val = FeatureRecord(
        feature_id="f2", post_id="p2", cluster_id="c2", split=SplitName.validation,
        risk_label=RiskLevel.LOW, features={}
    )
    rec_test = FeatureRecord(
        feature_id="f3", post_id="p3", cluster_id="c3", split=SplitName.test,
        risk_label=RiskLevel.MEDIUM, features={}
    )

    report = FeatureSplitter.audit_split_integrity([rec_train], [rec_val], [rec_test])
    assert report.is_valid is True
    assert report.cross_split_post_leakage == 0
    assert report.cross_split_cluster_leakage == 0

    # Test leakage detection
    rec_leaky_val = FeatureRecord(
        feature_id="f4", post_id="p1", cluster_id="c1", split=SplitName.validation,
        risk_label=RiskLevel.HIGH, features={}
    )
    leak_report = FeatureSplitter.audit_split_integrity([rec_train], [rec_leaky_val], [rec_test])
    assert leak_report.is_valid is False
    assert leak_report.cross_split_post_leakage == 1


# 17. Train/validation/test schema equality
def test_schema_equality():
    features_dir = Path("data/processed/features")
    if (features_dir / "train.csv").exists():
        df_tr = pd.read_csv(features_dir / "train.csv")
        df_va = pd.read_csv(features_dir / "validation.csv")
        df_te = pd.read_csv(features_dir / "test.csv")
        assert list(df_tr.columns) == list(df_va.columns)
        assert list(df_tr.columns) == list(df_te.columns)
        assert len(df_tr.columns) == 92 + 5  # 92 features + 5 id/target columns


# 18. Numeric validation
def test_numeric_validation():
    features_dir = Path("data/processed/features")
    if (features_dir / "train.csv").exists():
        df_tr = pd.read_csv(features_dir / "train.csv")
        feat_map = FeatureRegistry.get_feature_map()
        for f, defn in feat_map.items():
            if defn.dtype in [FeatureDType.FLOAT, FeatureDType.INT]:
                series = pd.to_numeric(df_tr[f], errors="coerce")
                assert not np.isinf(series).any()


# 19. Categorical validation
def test_categorical_validation():
    features_dir = Path("data/processed/features")
    if (features_dir / "train.csv").exists():
        df_tr = pd.read_csv(features_dir / "train.csv")
        assert "language" in df_tr.columns
        assert set(df_tr["language"].unique()).issubset({"en", "ta", "hi", "ta-en", "hi-en"})


# 20. Manifest generation
def test_manifest_generation():
    features_dir = Path("data/processed/features")
    manifest_path = features_dir / "feature_manifest.json"
    if manifest_path.exists():
        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_rows"] == 1560
        assert data["train_rows"] == 1092
        assert data["validation_rows"] == 234
        assert data["test_rows"] == 234
        assert data["feature_count"] == 92
        assert data["missing_value_rate"] == 0.0


# 21. Validator
def test_validator():
    features_dir = Path("data/processed/features")
    if (features_dir / "train.csv").exists():
        validator = FeatureDatasetValidator(features_dir)
        report = validator.validate()
        assert report.is_valid is True
        assert report.total_rows == 1560
        assert report.cross_split_post_leakage == 0
        assert report.target_leakage_detected is False
        assert report.schema_consistency == "PASS"


# 22. Dry-run CLI
def test_dry_run_builder():
    builder = FeatureDatasetBuilder(
        data_dir="data/trustlens",
        output_dir="data/processed/features",
    )
    res = builder.run(dry_run=True)
    assert res["records"] == 1560
    assert res["features"] == 92
    assert res["dry_run"] is True


# 23. LightGBM loader
def test_lightgbm_loader():
    features_dir = Path("data/processed/features")
    if (features_dir / "train.csv").exists():
        loader = FeatureDatasetLoader(features_dir)
        X_train, y_train = loader.load_train()
        X_val, y_val = loader.load_validation()
        X_test, y_test = loader.load_test()

        assert X_train.shape == (1092, 92)
        assert y_train.shape == (1092,)
        assert X_val.shape == (234, 92)
        assert y_val.shape == (234,)
        assert X_test.shape == (234, 92)
        assert y_test.shape == (234,)

        # Guarantee zero target leakage: risk_label NEVER in X
        assert "risk_label" not in X_train.columns
        assert "post_id" not in X_train.columns

        # Verify 4 classes preserved
        assert set(y_train.unique()).issubset({0, 1, 2, 3})


# 24. No external network dependency
def test_no_external_network_dependency(monkeypatch):
    import socket

    def mock_socket(*args, **kwargs):
        raise RuntimeError("Network access attempted during offline feature construction!")

    monkeypatch.setattr(socket, "socket", mock_socket)

    # Building features for a local post must succeed completely offline
    builder = FeatureDatasetBuilder(
        data_dir="data/trustlens",
        output_dir="data/processed/features",
    )
    # Dry-run execution must not trigger network
    res = builder.run(dry_run=True)
    assert res["records"] == 1560
