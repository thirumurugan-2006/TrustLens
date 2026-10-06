from typing import List, Dict, Any
from app.retrieval.schemas import CandidateEvidence
import uuid

class SemanticSearch:
    def retrieve(self, query_text: str, top_k: int = 5) -> List[CandidateEvidence]:
        return [
            CandidateEvidence(
                evidence_id=str(uuid.uuid4()),
                query_id="dummy_q",
                source_url="https://example.com/semantic",
                source_title="Semantic Result",
                source_type="LOCAL_DATA",
                snippet=f"Semantic match for {query_text}",
                retrieval_score=0.85,
                rank=1
            )
        ]
