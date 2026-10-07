from typing import List
from app.retrieval.schemas import CandidateEvidence
import logging

logger = logging.getLogger(__name__)

class SemanticSearch:
    def retrieve(self, query_text: str, top_k: int = 5) -> List[CandidateEvidence]:
        logger.warning("SemanticSearch is NOT_IMPLEMENTED. Returning empty list.")
        return []
