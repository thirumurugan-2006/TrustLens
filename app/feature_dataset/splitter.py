"""
TrustLens Cluster-Safe Feature Matrix Splitter (Phase 6D).
Partitions feature records into TRAIN, VALIDATION, and TEST sets while enforcing
strict cluster boundaries and zero cross-split leakage.
"""

from typing import Dict, List, Set, Tuple
from pydantic import BaseModel

from app.feature_dataset.schemas import FeatureRecord
from app.training.schemas import SplitName


class SplitIntegrityReport(BaseModel):
    """Integrity audit confirming cluster isolation across splits."""
    is_valid: bool
    train_count: int
    validation_count: int
    test_count: int
    total_count: int
    cross_split_post_leakage: int
    cross_split_cluster_leakage: int
    overlapping_post_ids: List[str]
    overlapping_cluster_ids: List[str]


class FeatureSplitter:
    """
    Partitions feature records by their authoritative cluster-safe split assignment.
    Verifies that neither post IDs nor cluster IDs cross split partitions.
    """

    @staticmethod
    def partition(records: List[FeatureRecord]) -> Tuple[List[FeatureRecord], List[FeatureRecord], List[FeatureRecord]]:
        """
        Partitions records into (train, validation, test) collections.
        """
        train_records: List[FeatureRecord] = []
        val_records: List[FeatureRecord] = []
        test_records: List[FeatureRecord] = []

        for rec in records:
            split_val = (rec.split.value if hasattr(rec.split, "value") else str(rec.split)).lower()
            if split_val in ("train", SplitName.train.value):
                train_records.append(rec)
            elif split_val in ("validation", "val", SplitName.validation.value):
                val_records.append(rec)
            elif split_val in ("test", SplitName.test.value):
                test_records.append(rec)
            else:
                raise ValueError(f"Unknown split name '{split_val}' on record {rec.feature_id}")

        return train_records, val_records, test_records

    @staticmethod
    def audit_split_integrity(
        train: List[FeatureRecord],
        val: List[FeatureRecord],
        test: List[FeatureRecord],
    ) -> SplitIntegrityReport:
        """
        Verifies absolute cluster-safe separation with zero post or cluster overlap.
        """
        train_posts: Set[str] = {r.post_id for r in train}
        val_posts: Set[str] = {r.post_id for r in val}
        test_posts: Set[str] = {r.post_id for r in test}

        post_leakage_tv = train_posts.intersection(val_posts)
        post_leakage_tt = train_posts.intersection(test_posts)
        post_leakage_vt = val_posts.intersection(test_posts)
        overlapping_posts = sorted(list(post_leakage_tv | post_leakage_tt | post_leakage_vt))

        train_clusters: Set[str] = {r.cluster_id for r in train if r.cluster_id}
        val_clusters: Set[str] = {r.cluster_id for r in val if r.cluster_id}
        test_clusters: Set[str] = {r.cluster_id for r in test if r.cluster_id}

        cluster_leakage_tv = train_clusters.intersection(val_clusters)
        cluster_leakage_tt = train_clusters.intersection(test_clusters)
        cluster_leakage_vt = val_clusters.intersection(test_clusters)
        overlapping_clusters = sorted(list(cluster_leakage_tv | cluster_leakage_tt | cluster_leakage_vt))

        is_valid = (len(overlapping_posts) == 0) and (len(overlapping_clusters) == 0)

        return SplitIntegrityReport(
            is_valid=is_valid,
            train_count=len(train),
            validation_count=len(val),
            test_count=len(test),
            total_count=len(train) + len(val) + len(test),
            cross_split_post_leakage=len(overlapping_posts),
            cross_split_cluster_leakage=len(overlapping_clusters),
            overlapping_post_ids=overlapping_posts,
            overlapping_cluster_ids=overlapping_clusters,
        )
