"""
TrustLens Retrieval Dataset Module (Phase 6B).
Constructs, manages, validates, and partitions dedicated retrieval triplets
(Query, Positive Evidence, Hard Negative Evidence) with cluster-aware zero leakage.
"""

# Lazy exports to avoid runpy RuntimeWarning when running python -m app.retrieval_dataset.validator / builder
from typing import TYPE_CHECKING

from app.retrieval_dataset.schemas import (
    NegativeType,
    NegativeEvidenceRecord,
    PositiveEvidenceRecord,
    RetrievalExample,
    RetrievalPair,
    RetrievalEvaluationItem,
)

if TYPE_CHECKING:
    from app.retrieval_dataset.query_builder import RetrievalQueryBuilder
    from app.retrieval_dataset.positive_selector import PositiveEvidenceSelector
    from app.retrieval_dataset.hard_negative_selector import HardNegativeSelector
    from app.retrieval_dataset.pair_builder import RetrievalPairBuilder
    from app.retrieval_dataset.cluster_manager import RetrievalClusterManager
    from app.retrieval_dataset.splitter import RetrievalDatasetSplitter
    from app.retrieval_dataset.validator import RetrievalDatasetValidator
    from app.retrieval_dataset.manifest import RetrievalDatasetManifest, RetrievalManifestBuilder

__all__ = [
    "NegativeType",
    "NegativeEvidenceRecord",
    "PositiveEvidenceRecord",
    "RetrievalExample",
    "RetrievalPair",
    "RetrievalEvaluationItem",
    "RetrievalQueryBuilder",
    "PositiveEvidenceSelector",
    "HardNegativeSelector",
    "RetrievalPairBuilder",
    "RetrievalClusterManager",
    "RetrievalDatasetSplitter",
    "RetrievalDatasetValidator",
    "RetrievalDatasetManifest",
    "RetrievalManifestBuilder",
]

def __getattr__(name: str):
    if name == "RetrievalQueryBuilder":
        from app.retrieval_dataset.query_builder import RetrievalQueryBuilder
        return RetrievalQueryBuilder
    elif name == "PositiveEvidenceSelector":
        from app.retrieval_dataset.positive_selector import PositiveEvidenceSelector
        return PositiveEvidenceSelector
    elif name == "HardNegativeSelector":
        from app.retrieval_dataset.hard_negative_selector import HardNegativeSelector
        return HardNegativeSelector
    elif name == "RetrievalPairBuilder":
        from app.retrieval_dataset.pair_builder import RetrievalPairBuilder
        return RetrievalPairBuilder
    elif name == "RetrievalClusterManager":
        from app.retrieval_dataset.cluster_manager import RetrievalClusterManager
        return RetrievalClusterManager
    elif name == "RetrievalDatasetSplitter":
        from app.retrieval_dataset.splitter import RetrievalDatasetSplitter
        return RetrievalDatasetSplitter
    elif name == "RetrievalDatasetValidator":
        from app.retrieval_dataset.validator import RetrievalDatasetValidator
        return RetrievalDatasetValidator
    elif name == "RetrievalDatasetManifest":
        from app.retrieval_dataset.manifest import RetrievalDatasetManifest
        return RetrievalDatasetManifest
    elif name == "RetrievalManifestBuilder":
        from app.retrieval_dataset.manifest import RetrievalManifestBuilder
        return RetrievalManifestBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

