from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class MultilingualRepresentation(BaseModel):
    """
    Semantic embedding representation of multilingual text.
    """
    text: str
    primary_language: str
    languages: List[str]
    scripts: List[str]
    is_code_mixed: bool
    is_transliteration: bool
    
    model_name: str
    embedding_dimension: int
    embedding: List[float]
    
    model_confidence: Optional[float] = None
    embedding_quality: str = "valid"  # "valid" or "invalid"
    
    metadata: Dict[str, Any] = {}
