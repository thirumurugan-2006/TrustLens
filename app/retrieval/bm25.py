from typing import List
from app.retrieval.schemas import CandidateEvidence
import logging

logger = logging.getLogger(__name__)

class BM25Retriever:
    def retrieve(self, query_text: str, top_k: int = 5) -> List[CandidateEvidence]:
        logger.warning("BM25Retriever is NOT_IMPLEMENTED. Returning empty list.")
        # Removed placeholder output to prevent false evidence generation.
        return []
