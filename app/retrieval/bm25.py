from typing import List, Dict, Any
from app.retrieval.schemas import CandidateEvidence
import uuid

class BM25Retriever:
    def retrieve(self, query_text: str, top_k: int = 5) -> List[CandidateEvidence]:
        # Minimal baseline BM25 mock for architecture completion
        return [
            CandidateEvidence(
                evidence_id=str(uuid.uuid4()),
                query_id="dummy_q",
                source_url="https://example.com/bm25",
                source_title="BM25 Result",
                source_type="LOCAL_DATA",
                snippet=f"BM25 match for {query_text}",
                retrieval_score=0.8,
                rank=1
            )
        ]
