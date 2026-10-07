"""
TrustLens Structured Feature Dataset Module (Phase 6D).
Builds, validates, serializes, and loads the structured feature matrix
for LightGBM and multimodal risk-model fusion.
"""

from typing import TYPE_CHECKING

from app.feature_dataset.schemas import (
    FeatureDefinition,
    FeatureDType,
    FeatureGroup,
    FeatureRecord,
    FeatureRegistry,
)

if TYPE_CHECKING:
    from app.feature_dataset.builder import FeatureDatasetBuilder
    from app.feature_dataset.completeness import FeatureCompletenessAnalyzer
    from app.feature_dataset.loader import (
        FeatureDatasetLoader,
        load_test,
        load_train,
        load_validation,
    )
    from app.feature_dataset.manifest import FeatureManifestBuilder
    from app.feature_dataset.serializer import FeatureSerializer
    from app.feature_dataset.splitter import FeatureSplitter
    from app.feature_dataset.validator import FeatureDatasetValidator

    from app.feature_dataset.claim_features import ClaimFeatureExtractor
    from app.feature_dataset.contradiction_features import ContradictionFeatureExtractor
    from app.feature_dataset.domain_features import DomainFeatureExtractor
    from app.feature_dataset.evidence_features import EvidenceFeatureExtractor
    from app.feature_dataset.graph_features import GraphFeatureExtractor
    from app.feature_dataset.image_features import ImageFeatureExtractor
    from app.feature_dataset.language_features import LanguageFeatureExtractor
    from app.feature_dataset.retrieval_features import RetrievalFeatureExtractor
    from app.feature_dataset.source_features import SourceFeatureExtractor
    from app.feature_dataset.text_features import TextFeatureExtractor


def __getattr__(name: str):
    if name == "FeatureDatasetBuilder":
        from app.feature_dataset.builder import FeatureDatasetBuilder
        return FeatureDatasetBuilder
    elif name == "FeatureDatasetValidator":
        from app.feature_dataset.validator import FeatureDatasetValidator
        return FeatureDatasetValidator
    elif name == "FeatureCompletenessAnalyzer":
        from app.feature_dataset.completeness import FeatureCompletenessAnalyzer
        return FeatureCompletenessAnalyzer
    elif name in ("FeatureDatasetLoader", "load_train", "load_validation", "load_test"):
        import app.feature_dataset.loader as ldr
        return getattr(ldr, name)
    elif name == "FeatureManifestBuilder":
        from app.feature_dataset.manifest import FeatureManifestBuilder
        return FeatureManifestBuilder
    elif name == "FeatureSerializer":
        from app.feature_dataset.serializer import FeatureSerializer
        return FeatureSerializer
    elif name == "FeatureSplitter":
        from app.feature_dataset.splitter import FeatureSplitter
        return FeatureSplitter
    elif name == "TextFeatureExtractor":
        from app.feature_dataset.text_features import TextFeatureExtractor
        return TextFeatureExtractor
    elif name == "LanguageFeatureExtractor":
        from app.feature_dataset.language_features import LanguageFeatureExtractor
        return LanguageFeatureExtractor
    elif name == "ClaimFeatureExtractor":
        from app.feature_dataset.claim_features import ClaimFeatureExtractor
        return ClaimFeatureExtractor
    elif name == "EvidenceFeatureExtractor":
        from app.feature_dataset.evidence_features import EvidenceFeatureExtractor
        return EvidenceFeatureExtractor
    elif name == "ContradictionFeatureExtractor":
        from app.feature_dataset.contradiction_features import ContradictionFeatureExtractor
        return ContradictionFeatureExtractor
    elif name == "RetrievalFeatureExtractor":
        from app.feature_dataset.retrieval_features import RetrievalFeatureExtractor
        return RetrievalFeatureExtractor
    elif name == "ImageFeatureExtractor":
        from app.feature_dataset.image_features import ImageFeatureExtractor
        return ImageFeatureExtractor
    elif name == "SourceFeatureExtractor":
        from app.feature_dataset.source_features import SourceFeatureExtractor
        return SourceFeatureExtractor
    elif name == "DomainFeatureExtractor":
        from app.feature_dataset.domain_features import DomainFeatureExtractor
        return DomainFeatureExtractor
    elif name == "GraphFeatureExtractor":
        from app.feature_dataset.graph_features import GraphFeatureExtractor
        return GraphFeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FeatureGroup",
    "FeatureDType",
    "FeatureDefinition",
    "FeatureRecord",
    "FeatureRegistry",
    "FeatureDatasetBuilder",
    "FeatureDatasetValidator",
    "FeatureSerializer",
    "FeatureSplitter",
    "FeatureCompletenessAnalyzer",
    "FeatureManifestBuilder",
    "FeatureDatasetLoader",
    "load_train",
    "load_validation",
    "load_test",
    "TextFeatureExtractor",
    "LanguageFeatureExtractor",
    "ClaimFeatureExtractor",
    "EvidenceFeatureExtractor",
    "ContradictionFeatureExtractor",
    "RetrievalFeatureExtractor",
    "ImageFeatureExtractor",
    "SourceFeatureExtractor",
    "DomainFeatureExtractor",
    "GraphFeatureExtractor",
]
