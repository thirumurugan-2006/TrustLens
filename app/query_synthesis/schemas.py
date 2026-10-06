from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid

class SearchQuery(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    atomic_claim_id: str
    query_text: str
    query_type: str
    language: str
    priority: str = "MEDIUM"
    rationale: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    source_preferences: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
