from typing import Any, Dict, List, Set, Tuple

from app.dataset.clusterer import DatasetClusterer
from app.training.schemas import ProvenanceSourceType, SplitMetadata, SplitName, TrainingPost


class DatasetSplitter:
    """
    Cluster-aware, leakage-safe dataset partitioner.
    Enforces:
    1. Indivisible cluster rule: Composite clusters move as single blocks.
    2. Synthetic exclusion: Records with source_type == SYNTHETIC or is_example == true
       are strictly prohibited from the test split.
    3. Target ratios: ~70% train, ~15% validation, ~15% test.
    """

    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_posts(
        self,
        posts: List[TrainingPost],
        clusterer: DatasetClusterer,
    ) -> Dict[str, SplitName]:
        """
        Assigns each post_id to a SplitName based on cluster grouping.
        Returns post_id -> SplitName mapping.
        """
        all_clusters = clusterer.get_all_clusters()  # cluster_id -> list of post_ids
        post_map = {p.post_id: p for p in posts}

        # Identify which clusters contain synthetic data
        synthetic_clusters: Set[str] = set()
        cluster_sizes: Dict[str, int] = {}

        for cluster_id, p_ids in all_clusters.items():
            cluster_sizes[cluster_id] = len(p_ids)
            for pid in p_ids:
                post = post_map.get(pid)
                if post and (post.is_example or post.provenance.source_type == ProvenanceSourceType.SYNTHETIC):
                    synthetic_clusters.add(cluster_id)
                    break

        total_records = len(posts)
        if total_records == 0:
            return {}

        target_test_records = int(total_records * self.test_ratio)
        target_val_records = int(total_records * self.val_ratio)

        cluster_split_assignment: Dict[str, SplitName] = {}

        # Sort clusters deterministically (by size descending, then cluster_id)
        sorted_clusters = sorted(all_clusters.keys(), key=lambda c: (-cluster_sizes[c], c))

        current_test_count = 0
        current_val_count = 0
        current_train_count = 0

        # Pass 1: Assign to test (ONLY non-synthetic clusters)
        for cluster_id in sorted_clusters:
            if cluster_id in synthetic_clusters:
                continue
            c_size = cluster_sizes[cluster_id]
            if current_test_count + c_size <= target_test_records or current_test_count == 0:
                cluster_split_assignment[cluster_id] = SplitName.test
                current_test_count += c_size
                if current_test_count >= target_test_records:
                    break

        # Pass 2: Assign to validation
        for cluster_id in sorted_clusters:
            if cluster_id in cluster_split_assignment:
                continue
            c_size = cluster_sizes[cluster_id]
            if current_val_count + c_size <= target_val_records or current_val_count == 0:
                cluster_split_assignment[cluster_id] = SplitName.validation
                current_val_count += c_size
                if current_val_count >= target_val_records:
                    break

        # Pass 3: Assign remaining clusters to train
        for cluster_id in sorted_clusters:
            if cluster_id not in cluster_split_assignment:
                cluster_split_assignment[cluster_id] = SplitName.train
                current_train_count += cluster_sizes[cluster_id]

        # Map back to post IDs
        post_split_map: Dict[str, SplitName] = {}
        for cluster_id, p_ids in all_clusters.items():
            assigned_split = cluster_split_assignment.get(cluster_id, SplitName.train)
            for pid in p_ids:
                post_split_map[pid] = assigned_split
                # Also update post.split_info if post is in post_map
                post = post_map.get(pid)
                if post:
                    c_info = clusterer.post_clusters.get(pid)
                    post.split_info = SplitMetadata(
                        split=assigned_split,
                        source_group_id=c_info.source_group_id if c_info else None,
                        campaign_group_id=c_info.campaign_group_id if c_info else None,
                        post_family_id=c_info.post_family_id if c_info else None,
                        image_family_id=c_info.image_family_id if c_info else None,
                        translation_group_id=c_info.translation_group_id if c_info else None,
                    )

        return post_split_map

    @classmethod
    def calculate_split_statistics(
        cls,
        posts: List[TrainingPost],
    ) -> Dict[str, Any]:
        """Computes summary statistics and percentage distribution across splits."""
        counts = {SplitName.train.value: 0, SplitName.validation.value: 0, SplitName.test.value: 0}
        total = len(posts)
        for p in posts:
            if p.split_info:
                counts[p.split_info.split.value] += 1
            else:
                counts[SplitName.train.value] += 1

        percentages = {k: round((v / total * 100), 2) if total > 0 else 0.0 for k, v in counts.items()}
        return {"counts": counts, "percentages": percentages, "total": total}
