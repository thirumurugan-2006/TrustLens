import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from app.claims.schemas import AtomicClaim, SourceSpan
from app.input.schemas import (
    AcquisitionMethod,
    AuthorInfo,
    Platform,
    PostContent,
    PostMedia,
    PostType,
    UniversalSocialPost,
)


class SplitName(str, Enum):
    train = "train"
    validation = "validation"
    test = "test"


class ProvenanceSourceType(str, Enum):
    PUBLIC_DATASET = "PUBLIC_DATASET"
    TRUSTLENS_ANNOTATED = "TRUSTLENS_ANNOTATED"
    SYNTHETIC = "SYNTHETIC"
    TRANSLATED = "TRANSLATED"
    HUMAN_REVIEWED = "HUMAN_REVIEWED"
    LOCAL_DATA = "LOCAL_DATA"
    USER_PROVIDED = "USER_PROVIDED"


class ClaimType(str, Enum):
    FINANCIAL = "FINANCIAL"
    JOB = "JOB"
    SHOPPING = "SHOPPING"
    GIVEAWAY = "GIVEAWAY"
    PAYMENT = "PAYMENT"
    CREDENTIAL = "CREDENTIAL"
    IMPERSONATION = "IMPERSONATION"
    OTHER = "OTHER"
    UNCLASSIFIED = "UNCLASSIFIED"


class ClaimDetectionLabel(str, Enum):
    CLAIM = "CLAIM"
    NON_CLAIM = "NON_CLAIM"
    UNCERTAIN = "UNCERTAIN"


class QueryType(str, Enum):
    ENTITY = "ENTITY"
    CLAIM = "CLAIM"
    SOURCE = "SOURCE"
    FINANCIAL = "FINANCIAL"
    TEMPORAL = "TEMPORAL"
    IMAGE = "IMAGE"
    VERIFICATION = "VERIFICATION"


class EvidenceRelationLabel(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"
    INSUFFICIENT = "INSUFFICIENT"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    INSUFFICIENT = "INSUFFICIENT"


class AnnotationConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    FLAGGED_FOR_REVIEW = "FLAGGED_FOR_REVIEW"


# ---------------------------------------------------------------------------
# Metadata Components
# ---------------------------------------------------------------------------


class ProvenanceMetadata(BaseModel):
    source_type: ProvenanceSourceType
    source_name: str
    source_id: Optional[str] = None
    collection_method: str = "manual"
    license: Optional[str] = None
    collection_date: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class LanguageMetadata(BaseModel):
    primary: str = "en"
    languages: List[str] = Field(default_factory=lambda: ["en"])
    script: List[str] = Field(default_factory=lambda: ["Latin"])
    code_mixed: bool = False
    transliterated: bool = False
    normalized_text: Optional[str] = None
    original_text: Optional[str] = None


class SplitMetadata(BaseModel):
    split: SplitName
    source_group_id: Optional[str] = None
    campaign_group_id: Optional[str] = None
    post_family_id: Optional[str] = None
    image_family_id: Optional[str] = None
    translation_group_id: Optional[str] = None


class ImageMetadata(BaseModel):
    image_id: str = Field(default_factory=lambda: f"img_{uuid.uuid4().hex[:10]}")
    image_path: Optional[str] = None
    image_hash: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_language: Optional[str] = None
    ocr_confidence: Optional[float] = None
    ocr_engine: Optional[str] = None
    image_quality: Optional[str] = None
    image_provenance: Optional[Dict[str, Any]] = None


class AnnotationMetadata(BaseModel):
    annotation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    annotator_id: str
    annotation_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: str = "1.0.0"
    guideline_version: str = "1.0.0"
    review_status: ReviewStatus = ReviewStatus.APPROVED
    confidence: AnnotationConfidence = AnnotationConfidence.HIGH
    notes: Optional[str] = None

    @model_validator(mode="after")
    def flag_low_confidence(self):
        if self.confidence == AnnotationConfidence.LOW and self.review_status == ReviewStatus.APPROVED:
            # Low confidence must be flagged for mandatory second review
            self.review_status = ReviewStatus.FLAGGED_FOR_REVIEW
        return self


# ---------------------------------------------------------------------------
# Training Records
# ---------------------------------------------------------------------------


class TrainingPost(UniversalSocialPost):
    language_info: LanguageMetadata = Field(default_factory=LanguageMetadata)
    provenance: ProvenanceMetadata
    split_info: Optional[SplitMetadata] = None
    images_meta: List[ImageMetadata] = Field(default_factory=list)
    is_example: bool = False
    schema_version: str = "1.0.0"


class TrainingClaim(BaseModel):
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    post_id: str
    claim_text: str
    claim_type: ClaimType = ClaimType.UNCLASSIFIED
    detection_label: ClaimDetectionLabel = ClaimDetectionLabel.CLAIM
    claim_status: str = "ACTIVE"
    check_worthiness: float = Field(ge=0.0, le=1.0, default=1.0)
    source_span: Optional[SourceSpan] = None
    language: str = "en"
    script: str = "Latin"
    attribution: Optional[Dict[str, Any]] = None
    modality: Optional[str] = None
    provenance: ProvenanceMetadata
    split_info: Optional[SplitMetadata] = None
    annotation: Optional[AnnotationMetadata] = None
    is_example: bool = False


class TrainingAtomicClaim(AtomicClaim):
    claim_type_enum: Optional[ClaimType] = None
    provenance: Optional[ProvenanceMetadata] = None
    split_info: Optional[SplitMetadata] = None
    annotation: Optional[AnnotationMetadata] = None
    is_example: bool = False

    @model_validator(mode="before")
    @classmethod
    def sync_claim_type(cls, data: Any) -> Any:
        if isinstance(data, dict):
            c_type = data.get("claim_type")
            if c_type and isinstance(c_type, str):
                c_upper = c_type.upper()
                if c_upper in ClaimType.__members__:
                    data["claim_type_enum"] = ClaimType[c_upper]
        return data


class TrainingQuery(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str
    atomic_claim_id: Optional[str] = None
    query_text: str
    query_type: QueryType = QueryType.VERIFICATION
    language: str = "en"
    generation_method: str = "rule_based"
    provenance: ProvenanceMetadata
    is_example: bool = False


class TrainingEvidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str
    evidence_text: str
    source_url: Optional[str] = None
    source_type: str = "WEB_DOCUMENT"
    source_title: Optional[str] = None
    retrieval_method: str = "bm25_dense"
    retrieval_query_id: Optional[str] = None
    source_timestamp: Optional[str] = None
    language: str = "en"
    relation_label: EvidenceRelationLabel
    provenance: ProvenanceMetadata
    reliability_metadata: Dict[str, Any] = Field(default_factory=dict)
    annotation: Optional[AnnotationMetadata] = None
    is_example: bool = False


class TrainingRisk(BaseModel):
    risk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    post_id: str
    claim_ids: List[str] = Field(default_factory=list)
    risk_level: RiskLevel
    risk_factors: List[str] = Field(default_factory=list)
    evidence_summary: str = ""
    confidence: AnnotationConfidence = AnnotationConfidence.HIGH
    provenance: ProvenanceMetadata
    annotation: Optional[AnnotationMetadata] = None
    is_example: bool = False


class DatasetManifest(BaseModel):
    dataset_id: str
    dataset_version: str
    schema_version: str = "1.0.0"
    guideline_version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_summary: str
    license: str
    record_count: int
    language_distribution: Dict[str, int]
    label_distribution: Dict[str, int]
    splits_distribution: Dict[str, int]
    leakage_group_counts: Dict[str, int] = Field(default_factory=dict)
