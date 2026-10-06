from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class EvidenceStance(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"
    INSUFFICIENT = "INSUFFICIENT"

class Evidence(BaseModel):
    evidence_id: str
    atomic_claim_id: str
    source_url: str
    source_title: str
    source_type: str
    snippet: str
    stance: EvidenceStance
    relevance_score: float
    source_reliability: float
    provenance: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
