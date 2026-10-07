from typing import Any, Dict, List

from app.retrieval_dataset.schemas import RetrievalExample, RetrievalPair
from app.training.schemas import SplitName


class RetrievalDatasetSplitter:
    """
    Partitions retrieval examples and pairs into Train, Validation, and Test sets
    strictly adhering to cluster-safe split assignments.
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
        """
        Deterministically assigns cluster IDs to train, validation, and test splits.
        """
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
    def partition_examples(
        examples: List[RetrievalExample],
    ) -> Dict[str, List[RetrievalExample]]:
        """Splits RetrievalExamples into train, validation, and test subsets."""
        splits: Dict[str, List[RetrievalExample]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for ex in examples:
            s_name = ex.split.value if hasattr(ex.split, "value") else str(ex.split)
            if s_name in splits:
                splits[s_name].append(ex)
            else:
                splits["train"].append(ex)
        return splits

    @staticmethod
    def partition_pairs(
        pairs: List[RetrievalPair],
    ) -> Dict[str, List[RetrievalPair]]:
        """Splits RetrievalPairs into train, validation, and test subsets."""
        splits: Dict[str, List[RetrievalPair]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for p in pairs:
            s_name = p.split.value if hasattr(p.split, "value") else str(p.split)
            if s_name in splits:
                splits[s_name].append(p)
            else:
                splits["train"].append(p)
        return splits
