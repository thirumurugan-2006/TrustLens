from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from app.retrieval_dataset.schemas import RetrievalExample, RetrievalPair


class RetrievalDatasetManifest(BaseModel):
    """
    Dataset manifest capturing statistics, distributions, and integrity metrics
    for the TrustLens Retrieval and Reranking Dataset.
    """
    dataset_id: str = "trustlens_retrieval_v0.1.0"
    dataset_version: str = "v0.1.0"
    source_dataset_version: str = "v0.2.0"
    schema_version: str = "1.0.0"
    guideline_version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Record counts
    record_count: int = 0  # Total RetrievalExamples
    pair_count: int = 0    # Total RetrievalPairs (positive + negative)
    query_count: int = 0
    claim_count: int = 0
    positive_count: int = 0
    negative_count: int = 0

    # Distributions
    negative_type_distribution: Dict[str, int] = Field(default_factory=dict)
    positive_relation_distribution: Dict[str, int] = Field(default_factory=dict)
    query_type_distribution: Dict[str, int] = Field(default_factory=dict)
    language_distribution: Dict[str, int] = Field(default_factory=dict)
    cross_language_count: int = 0

    # Splits
    train_count: int = 0
    validation_count: int = 0
    test_count: int = 0

    # Integrity & Provenance
    source_distribution: Dict[str, int] = Field(default_factory=dict)
    cluster_count: int = 0
    duplicate_count: int = 0
    invalid_count: int = 0
    human_reviewed_count: int = 0


class RetrievalManifestBuilder:
    """Computes manifest statistics from RetrievalExamples and RetrievalPairs."""

    @classmethod
    def build_manifest(
        cls,
        examples: List[RetrievalExample],
        pairs: Optional[List[RetrievalPair]] = None,
        duplicate_count: int = 0,
        invalid_count: int = 0,
        dataset_version: str = "v0.1.0",
        source_dataset_version: str = "v0.2.0",
    ) -> RetrievalDatasetManifest:
        """Computes accurate manifest counts from generated retrieval records."""
        unique_queries = {ex.query_id for ex in examples}
        unique_claims = {ex.claim_id for ex in examples}
        unique_positives = {ex.positive_evidence_id for ex in examples}
        unique_clusters = {ex.cluster_id for ex in examples}

        neg_type_dist: Dict[str, int] = {}
        pos_rel_dist: Dict[str, int] = {}
        q_type_dist: Dict[str, int] = {}
        lang_dist: Dict[str, int] = {}
        source_dist: Dict[str, int] = {}
        split_counts: Dict[str, int] = {"train": 0, "validation": 0, "test": 0}

        cross_lang_count = 0
        total_negatives = 0
        human_reviewed_count = 0

        for ex in examples:
            # Positive relation
            p_rel = ex.positive_relation.value if hasattr(ex.positive_relation, "value") else str(ex.positive_relation)
            pos_rel_dist[p_rel] = pos_rel_dist.get(p_rel, 0) + 1

            # Query type
            qt = ex.query_type.value if hasattr(ex.query_type, "value") else str(ex.query_type)
            q_type_dist[qt] = q_type_dist.get(qt, 0) + 1

            # Language
            qlang = ex.query_language or ex.language or "en"
            lang_dist[qlang] = lang_dist.get(qlang, 0) + 1

            # Cross-language
            if ex.cross_language:
                cross_lang_count += 1

            # Split
            s_name = ex.split.value if hasattr(ex.split, "value") else str(ex.split)
            split_counts[s_name] = split_counts.get(s_name, 0) + 1

            # Source
            stype = ex.provenance.source_type.value if ex.provenance else "UNKNOWN"
            source_dist[stype] = source_dist.get(stype, 0) + 1

            # Negatives
            for neg in ex.negatives:
                total_negatives += 1
                nt = neg.negative_type.value if hasattr(neg.negative_type, "value") else str(neg.negative_type)
                neg_type_dist[nt] = neg_type_dist.get(nt, 0) + 1

            # Human review
            if ex.human_reviewed:
                human_reviewed_count += 1

        return RetrievalDatasetManifest(
            dataset_version=dataset_version,
            source_dataset_version=source_dataset_version,
            record_count=len(examples),
            pair_count=len(pairs) if pairs else 0,
            query_count=len(unique_queries),
            claim_count=len(unique_claims),
            positive_count=len(unique_positives),
            negative_count=total_negatives,
            negative_type_distribution=neg_type_dist,
            positive_relation_distribution=pos_rel_dist,
            query_type_distribution=q_type_dist,
            language_distribution=lang_dist,
            cross_language_count=cross_lang_count,
            train_count=split_counts.get("train", 0),
            validation_count=split_counts.get("validation", 0),
            test_count=split_counts.get("test", 0),
            source_distribution=source_dist,
            cluster_count=len(unique_clusters),
            duplicate_count=duplicate_count,
            invalid_count=invalid_count,
            human_reviewed_count=human_reviewed_count,
        )

    @classmethod
    def save_manifest(
        cls,
        manifest: RetrievalDatasetManifest,
        output_path: Union[str, Path] = "data/processed/retrieval/dataset_manifest.json",
    ) -> Path:
        """Saves manifest JSON to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path
