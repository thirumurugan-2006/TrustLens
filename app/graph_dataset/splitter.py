"""
TrustLens Graph Splitter (Phase 6C).
Partitions heterogeneous evidence graphs into Train, Validation, and Test splits by cluster.
"""

from typing import Any, Dict, List

from app.graph_dataset.schemas import (
    HeterogeneousEvidenceGraph,
)
from app.training.schemas import SplitName


class GraphSplitter:
    """
    Partitions graphs into Train, Validation, and Test subsets adhering strictly to cluster boundaries.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_clusters(
        self,
        clusters: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """Deterministically assigns cluster IDs to train, validation, and test splits."""
        sorted_clusters = sorted(clusters.keys())
        total = len(sorted_clusters)
        n_train = int(total * self.train_ratio)
        n_val = int(total * self.val_ratio)

        train_clusters = sorted_clusters[:n_train]
        val_clusters = sorted_clusters[n_train : n_train + n_val]
        test_clusters = sorted_clusters[n_train + n_val :]

        return {
            "train": train_clusters,
            "validation": val_clusters,
            "test": test_clusters,
        }

    @staticmethod
    def partition_graphs(
        graphs: List[HeterogeneousEvidenceGraph],
    ) -> Dict[str, List[HeterogeneousEvidenceGraph]]:
        """Splits HeterogeneousEvidenceGraph instances into train, validation, and test lists."""
        splits: Dict[str, List[HeterogeneousEvidenceGraph]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for g in graphs:
            s_name = g.split.value if hasattr(g.split, "value") else str(g.split)
            if s_name in splits:
                splits[s_name].append(g)
            else:
                splits["train"].append(g)
        return splits
