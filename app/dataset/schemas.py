import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from app.claims.schemas import AtomicClaim, Claim
from app.input.schemas import Platform, PostType, UniversalSocialPost
from app.query_synthesis.schemas import SearchQuery
from app.training.schemas import (
    AnnotationConfidence,
    AnnotationMetadata,
    ClaimDetectionLabel,
    ClaimType,
    EvidenceRelationLabel,
    LanguageMetadata,
    ProvenanceMetadata,
    ProvenanceSourceType,
    RiskLevel,
    SplitMetadata,
    SplitName,
    TrainingPost,
)


class AnnotationTaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUBMITTED = "SUBMITTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    ADJUDICATION = "ADJUDICATION"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class DuplicateType(str, Enum):
    NONE = "NONE"
    EXACT = "EXACT"
    TEXT_SIMILARITY = "TEXT_SIMILARITY"
    IMAGE_PHASH = "IMAGE_PHASH"
    CROSS_LINGUAL_VARIANT = "CROSS_LINGUAL_VARIANT"


class RawDataRecord(BaseModel):
    raw_id: str = Field(default_factory=lambda: f"raw_{uuid.uuid4().hex[:12]}")
    format: str  # json, jsonl, csv, txt
    raw_payload: Dict[str, Any]
    source_path: Optional[str] = None
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DuplicateInfo(BaseModel):
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    duplicate_type: DuplicateType = DuplicateType.NONE
    duplicate_score: float = 0.0


class ClusterInfo(BaseModel):
    source_group_id: Optional[str] = None
    campaign_group_id: Optional[str] = None
    post_family_id: Optional[str] = None
    image_family_id: Optional[str] = None
    translation_group_id: Optional[str] = None
    composite_cluster_id: str = Field(default_factory=lambda: f"clus_{uuid.uuid4().hex[:10]}")


class SemanticAutoSuggestions(BaseModel):
    claims: List[Claim] = Field(default_factory=list)
    atomic_claims: List[AtomicClaim] = Field(default_factory=list)
    search_queries: List[SearchQuery] = Field(default_factory=list)
    is_auto_generated: bool = True
    generation_pipeline: str = "TrustLensClaimPipeline_v1"


class HumanAnnotationSubmission(BaseModel):
    annotator_id: str
    submitted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    claim_detection: Optional[ClaimDetectionLabel] = None
    claim_type: Optional[ClaimType] = None
    atomic_frames: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_relations: List[Dict[str, Any]] = Field(default_factory=list)
    risk_level: Optional[RiskLevel] = None
    risk_factors: List[str] = Field(default_factory=list)
    evidence_summary: Optional[str] = None
    confidence: AnnotationConfidence = AnnotationConfidence.HIGH
    notes: Optional[str] = None


class AnnotationTask(BaseModel):
    annotation_task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    post_id: str
    post: TrainingPost
    status: AnnotationTaskStatus = AnnotationTaskStatus.PENDING
    priority: int = 1  # 1 = normal, 2 = high, 3 = critical
    assigned_annotator: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    auto_suggestions: Optional[SemanticAutoSuggestions] = None
    submissions: List[HumanAnnotationSubmission] = Field(default_factory=list)
    consensus_submission: Optional[HumanAnnotationSubmission] = None
    qc_notes: List[str] = Field(default_factory=list)


class QCResult(BaseModel):
    passed: bool
    status: str  # ACCEPTED or REJECTED
    reason_codes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class DatasetValidationReport(BaseModel):
    valid: bool
    total_records: int
    schema_valid: bool
    provenance_valid: bool
    duplicates_detected: int
    clusters_count: int
    train_count: int
    validation_count: int
    test_count: int
    synthetic_in_test_violations: int
    cross_split_leakage_violations: int
    language_distribution: Dict[str, int]
    risk_distribution: Dict[str, int]
    claim_distribution: Dict[str, int]
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
