from typing import List
from app.retrieval.schemas import CandidateEvidence

class Reranker:
    def rerank(self, candidates: List[CandidateEvidence]) -> List[CandidateEvidence]:
        # Currently a pass-through placeholder. Not implemented.
        return sorted(candidates, key=lambda x: x.retrieval_score, reverse=True) if candidates else []
