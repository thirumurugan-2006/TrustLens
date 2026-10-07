from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from app.claims.schemas import SourceSpan, AtomicClaim, DecomposedClaim

class Source(BaseModel):
    source_id: str
    name: str
    url: Optional[str] = None
    domain: Optional[str] = None
    reliability_score: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Evidence(BaseModel):
    evidence_id: str
    claim_id: str
    text: str
    source: Source
    url: Optional[str] = None
    stance: str = "NEUTRAL" # SUPPORTS, CONTRADICTS, IRRELEVANT, INSUFFICIENT
    relevance_score: float = 0.0
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Claim(BaseModel):
    claim_id: str
    text: str
    entities: List[str] = Field(default_factory=list)
    language: str = "unknown"
    confidence: float = 1.0
    claim_type: str = "UNKNOWN"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NormalizedPost(BaseModel):
    platform: str
    post_id: str
    url: Optional[str] = None
    author: Optional[str] = None
    text: str
    title: Optional[str] = None
    description: Optional[str] = None
    images: List[str] = Field(default_factory=list)
    videos: List[str] = Field(default_factory=list)
    timestamp: Optional[str] = None
    comments: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GraphNode(BaseModel):
    node_id: str
    node_type: str # CLAIM, EVIDENCE, SOURCE, ENTITY
    attributes: Dict[str, Any] = Field(default_factory=dict)

class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: str # SUPPORTED_BY, CONTRADICTED_BY, FROM, ABOUT

class EvidenceGraph(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)

class RiskResult(BaseModel):
    raw_risk_score: float
    risk_class: str # LOW_RISK, MEDIUM_RISK, HIGH_RISK, INSUFFICIENT_EVIDENCE
    calibrated_probability: Optional[float] = None
    components: Dict[str, Any] = Field(default_factory=dict)

class TrustLensResult(BaseModel):
    post_id: str
    risk_level: str
    confidence: float
    claims: List[Claim] = Field(default_factory=list)
    supporting_evidence: List[Evidence] = Field(default_factory=list)
    contradicting_evidence: List[Evidence] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    reasoning_signals: Dict[str, Any] = Field(default_factory=dict)
    abstained: bool = False
    explanation: Optional[str] = None
