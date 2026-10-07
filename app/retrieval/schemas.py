from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class CandidateEvidence(BaseModel):
    evidence_id: str
    query_id: str
    source_url: str
    source_title: str
    source_type: str  # LIVE_WEB, LOCAL_CACHE, LOCAL_DATA
    snippet: str
    retrieval_score: float
    rank: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
