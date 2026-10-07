from app.training.schemas import (
    AnnotationConfidence,
    AnnotationMetadata,
    ClaimDetectionLabel,
    ClaimType,
    DatasetManifest,
    EvidenceRelationLabel,
    ImageMetadata,
    LanguageMetadata,
    ProvenanceMetadata,
    ProvenanceSourceType,
    QueryType,
    ReviewStatus,
    RiskLevel,
    SplitMetadata,
    SplitName,
    TrainingAtomicClaim,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingQuery,
    TrainingRisk,
)

__all__ = [
    "SplitName",
    "ProvenanceSourceType",
    "ClaimType",
    "ClaimDetectionLabel",
    "QueryType",
    "EvidenceRelationLabel",
    "RiskLevel",
    "AnnotationConfidence",
    "ReviewStatus",
    "ProvenanceMetadata",
    "LanguageMetadata",
    "SplitMetadata",
    "ImageMetadata",
    "AnnotationMetadata",
    "TrainingPost",
    "TrainingClaim",
    "TrainingAtomicClaim",
    "TrainingQuery",
    "TrainingEvidence",
    "TrainingRisk",
    "DatasetManifest",
    "EvaluationFramework",
    "BaselineFramework",
    "MajorityClassBaseline",
    "TfidfLogisticRegressionBaseline",
    "TfidfLinearSVMBaseline",
    "TrainingReadinessAuditor",
]


def __getattr__(name: str):
    if name == "EvaluationFramework":
        from app.training.evaluation import EvaluationFramework
        return EvaluationFramework
    elif name in ("BaselineFramework", "MajorityClassBaseline", "TfidfLogisticRegressionBaseline", "TfidfLinearSVMBaseline"):
        import app.training.baselines as baselines_mod
        return getattr(baselines_mod, name)
    elif name == "TrainingReadinessAuditor":
        from app.training.readiness_audit import TrainingReadinessAuditor
        return TrainingReadinessAuditor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
