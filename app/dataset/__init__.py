"""
TrustLens Phase 4A Dataset Infrastructure:
Data Collection, Normalization, Deduplication, Annotation Queue, Quality Control, and Validation.
"""

from typing import Any

__all__ = [
    "AnnotationQueue",
    "AnnotationQueueError",
    "AnnotationTask",
    "AnnotationTaskStatus",
    "ClusterInfo",
    "DatasetClusterer",
    "DatasetCollector",
    "DatasetDeduplicator",
    "DatasetManifestBuilder",
    "DatasetNormalizer",
    "DatasetQualityControl",
    "DatasetSplitter",
    "DatasetValidationReport",
    "DatasetValidator",
    "DisjointSetUnion",
    "DuplicateInfo",
    "DuplicateType",
    "HumanAnnotationSubmission",
    "NormalizationError",
    "Phase4DatasetManifest",
    "ProvenanceError",
    "ProvenanceManager",
    "QCResult",
    "RawDataRecord",
    "SemanticAutoSuggestions",
    "SourceEntry",
    "SourceRegistry",
    "DevelopmentDatasetBuilder",
]

_MODULE_MAP = {
    "AnnotationQueue": "app.dataset.annotation_queue",
    "AnnotationQueueError": "app.dataset.annotation_queue",
    "AnnotationTask": "app.dataset.schemas",
    "AnnotationTaskStatus": "app.dataset.schemas",
    "ClusterInfo": "app.dataset.schemas",
    "DatasetClusterer": "app.dataset.clusterer",
    "DatasetCollector": "app.dataset.collector",
    "DatasetDeduplicator": "app.dataset.deduplicator",
    "DatasetManifestBuilder": "app.dataset.manifest",
    "DatasetNormalizer": "app.dataset.normalizer",
    "DatasetQualityControl": "app.dataset.quality_control",
    "DatasetSplitter": "app.dataset.splitter",
    "DatasetValidationReport": "app.dataset.schemas",
    "DatasetValidator": "app.dataset.validator",
    "DevelopmentDatasetBuilder": "app.dataset.builder",
    "DisjointSetUnion": "app.dataset.clusterer",
    "DuplicateInfo": "app.dataset.schemas",
    "DuplicateType": "app.dataset.schemas",
    "HumanAnnotationSubmission": "app.dataset.schemas",
    "NormalizationError": "app.dataset.normalizer",
    "Phase4DatasetManifest": "app.dataset.manifest",
    "ProvenanceError": "app.dataset.provenance",
    "ProvenanceManager": "app.dataset.provenance",
    "QCResult": "app.dataset.schemas",
    "RawDataRecord": "app.dataset.schemas",
    "SemanticAutoSuggestions": "app.dataset.schemas",
    "SourceEntry": "app.dataset.source_registry",
    "SourceRegistry": "app.dataset.source_registry",
}


def __getattr__(name: str) -> Any:
    if name in _MODULE_MAP:
        import importlib

        mod = importlib.import_module(_MODULE_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
