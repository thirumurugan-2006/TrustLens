import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from app.training.schemas import (
    ClaimDetectionLabel,
    ClaimType,
    EvidenceRelationLabel,
    ProvenanceSourceType,
    RiskLevel,
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingRisk,
)


class Phase4DatasetManifest(BaseModel):
    """
    Dataset manifest capturing distributions, counts, provenance, and cluster statistics
    in strict adherence to the TrustLens dataset specification.
    """
    dataset_id: str = "trustlens_phase4a"
    dataset_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    guideline_version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Metadata & Legal
    source_summary: str = "TrustLens multilingual multimodal scam-risk dataset"
    license: str = "Research & Evaluation - Permitted / Fair Use"

    # Core Counts
    record_count: int = 0
    total_records: int = 0
    train_count: int = 0
    validation_count: int = 0
    test_count: int = 0

    # Categorical Distributions
    language_distribution: Dict[str, int] = Field(default_factory=dict)
    claim_distribution: Dict[str, int] = Field(default_factory=dict)
    risk_distribution: Dict[str, int] = Field(default_factory=dict)
    evidence_distribution: Dict[str, int] = Field(default_factory=dict)
    category_distribution: Dict[str, int] = Field(default_factory=dict)
    source_distribution: Dict[str, int] = Field(default_factory=dict)

    # Synthetic vs Real
    synthetic_count: int = 0
    real_count: int = 0

    # Pipeline Integrity Metrics
    duplicate_count: int = 0
    rejected_count: int = 0
    cluster_count: int = 0

    # Annotation Tracking Metrics
    annotation_count: int = 0
    double_annotated_count: int = 0
    adjudicated_count: int = 0

    # Schema-compliant convenience fields
    label_distribution: Dict[str, int] = Field(default_factory=dict)
    splits_distribution: Dict[str, int] = Field(default_factory=dict)
    leakage_group_counts: Dict[str, int] = Field(default_factory=dict)
    domain_distribution: Dict[str, int] = Field(default_factory=dict)

    # Versioning & Phase 6A Remediation fields
    parent_version: Optional[str] = None
    medium_count: int = 0
    neutral_count: int = 0
    synthetic_test_count: int = 0
    new_records: int = 0
    removed_records: int = 0
    modified_records: int = 0


class DatasetManifestBuilder:
    """
    Builds and writes data/trustlens/dataset_manifest.json from real processed posts.
    """

    @classmethod
    def build_manifest(
        cls,
        posts: List[TrainingPost],
        claims: Optional[List[TrainingClaim]] = None,
        evidence: Optional[List[TrainingEvidence]] = None,
        risks: Optional[List[TrainingRisk]] = None,
        category_distribution: Optional[Dict[str, int]] = None,
        annotation_count: Optional[int] = None,
        double_annotated_count: int = 0,
        adjudicated_count: int = 0,
        duplicate_count: int = 0,
        rejected_count: int = 0,
        cluster_count: int = 0,
        dataset_id: str = "trustlens_phase4a",
        dataset_version: str = "1.0.0",
        license_str: str = "Research & Evaluation - Permitted / Fair Use",
        parent_version: Optional[str] = None,
        new_records: int = 0,
        removed_records: int = 0,
        modified_records: int = 0,
    ) -> Phase4DatasetManifest:
        """
        Computes accurate statistics from input training records.
        """
        lang_dist: Dict[str, int] = {}
        src_dist: Dict[str, int] = {}
        split_counts: Dict[str, int] = {"train": 0, "validation": 0, "test": 0}
        leakage_group_counts: Dict[str, int] = {}

        synthetic_count = 0
        real_count = 0
        synthetic_test_count = 0

        post_categories: Dict[str, int] = {}

        for post in posts:
            # Language
            lang = post.language_info.primary if post.language_info else "unknown"
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

            # Source
            stype = post.provenance.source_type.value if post.provenance else "UNKNOWN"
            src_dist[stype] = src_dist.get(stype, 0) + 1

            # Synthetic vs Real
            is_synth = post.is_example or (post.provenance and post.provenance.source_type == ProvenanceSourceType.SYNTHETIC)
            if is_synth:
                synthetic_count += 1
            else:
                real_count += 1

            # Category from post metadata if present
            cat = post.metadata.get("category")
            if cat:
                post_categories[str(cat).upper()] = post_categories.get(str(cat).upper(), 0) + 1

            # Split
            if post.split_info and post.split_info.split:
                sname = post.split_info.split.value
                split_counts[sname] = split_counts.get(sname, 0) + 1

                if is_synth and sname == "test":
                    synthetic_test_count += 1

                # Leakage groups
                if post.split_info.campaign_group_id:
                    leakage_group_counts["campaign"] = leakage_group_counts.get("campaign", 0) + 1
                if post.split_info.translation_group_id:
                    leakage_group_counts["translation"] = leakage_group_counts.get("translation", 0) + 1
                if post.split_info.post_family_id:
                    leakage_group_counts["post_family"] = leakage_group_counts.get("post_family", 0) + 1

        # Claim distribution
        claim_dist: Dict[str, int] = {}
        if claims:
            for c in claims:
                ctype = c.claim_type.value if isinstance(c.claim_type, ClaimType) else str(c.claim_type)
                claim_dist[ctype] = claim_dist.get(ctype, 0) + 1

        # Risk distribution
        risk_dist: Dict[str, int] = {}
        if risks:
            for r in risks:
                rlevel = r.risk_level.value if isinstance(r.risk_level, RiskLevel) else str(r.risk_level)
                risk_dist[rlevel] = risk_dist.get(rlevel, 0) + 1

        # Evidence distribution
        ev_dist: Dict[str, int] = {}
        if evidence:
            for e in evidence:
                erel = e.relation_label.value if isinstance(e.relation_label, EvidenceRelationLabel) else str(e.relation_label)
                ev_dist[erel] = ev_dist.get(erel, 0) + 1

        # Combined label distribution
        label_dist: Dict[str, int] = {}
        for k, v in risk_dist.items():
            label_dist[f"risk_{k}"] = v
        for k, v in claim_dist.items():
            label_dist[f"claim_{k}"] = v

        final_category_dist = category_distribution if category_distribution is not None else (post_categories or claim_dist)
        final_ann_count = annotation_count if annotation_count is not None else (len(claims) if claims else len(posts))

        return Phase4DatasetManifest(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            license=license_str,
            record_count=len(posts),
            total_records=len(posts),
            train_count=split_counts.get("train", 0),
            validation_count=split_counts.get("validation", 0),
            test_count=split_counts.get("test", 0),
            language_distribution=lang_dist,
            claim_distribution=claim_dist,
            risk_distribution=risk_dist,
            evidence_distribution=ev_dist,
            category_distribution=final_category_dist,
            domain_distribution=final_category_dist,
            source_distribution=src_dist,
            synthetic_count=synthetic_count,
            real_count=real_count,
            duplicate_count=duplicate_count,
            rejected_count=rejected_count,
            cluster_count=cluster_count,
            annotation_count=final_ann_count,
            double_annotated_count=double_annotated_count,
            adjudicated_count=adjudicated_count,
            label_distribution=label_dist,
            splits_distribution=split_counts,
            leakage_group_counts=leakage_group_counts,
            parent_version=parent_version,
            medium_count=risk_dist.get("MEDIUM", 0),
            neutral_count=ev_dist.get("NEUTRAL", 0),
            synthetic_test_count=synthetic_test_count,
            new_records=new_records,
            removed_records=removed_records,
            modified_records=modified_records,
        )

    @classmethod
    def save_manifest(
        cls,
        manifest: Phase4DatasetManifest,
        output_path: Union[str, Path] = "data/trustlens/dataset_manifest.json",
    ) -> Path:
        """Saves dataset manifest to specified destination."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path
