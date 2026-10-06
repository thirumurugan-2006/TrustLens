from app.retrieval.schemas import CandidateEvidence
from typing import List

class Reranker:
    def rerank(self, candidates: List[CandidateEvidence]) -> List[CandidateEvidence]:
        # Dummy reranker
        return sorted(candidates, key=lambda x: x.retrieval_score, reverse=True)
