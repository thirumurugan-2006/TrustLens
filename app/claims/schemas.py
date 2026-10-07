from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import uuid

class SourceSpan(BaseModel):
    start: int
    end: int
    source_sentence: str
    source_index: int

class Claim(BaseModel):
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    post_id: Optional[str] = None
    text: str
    normalized_text: str
    claim_type: str = "UNKNOWN"
    language: str = "unknown"
    source_span: Optional[SourceSpan] = None
    extraction_method: str = "rule_based"
    extraction_confidence: float = 1.0
    verifiable: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AtomicClaim(BaseModel):
    atomic_claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_claim_id: str
    original_text: str
    normalized_text: str
    claim_type: str = "UNKNOWN"
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    currency: Optional[str] = None
    temporal_context: Optional[Dict[str, Any]] = None
    polarity: str = "POSITIVE"
    negated: bool = False
    entities: List[str] = Field(default_factory=list)
    conditions: Optional[Dict[str, Any]] = None
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    attribution: Optional[Dict[str, Any]] = None
    modality: Optional[str] = None
    certainty: Optional[str] = None
    language: str = "unknown"
    script: str = "unknown"
    source_span: Optional[SourceSpan] = None
    verifiable: bool = False
    decomposition_confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DecomposedClaim(BaseModel):
    parent_claim_id: str
    atomic_claims: List[AtomicClaim]
    decomposition_status: str = "success"
    decomposition_confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
