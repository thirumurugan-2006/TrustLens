"""
TrustLens Retrieval Feature Extractor (Phase 6D).
Extracts grounded retrieval and candidate pool features from Phase 6B outputs.
Strict rule: Never use hard-negative count as a direct scam risk feature.
"""

from typing import Any, Dict, List, Optional


class RetrievalFeatureExtractor:
    """Extracts retrieval candidate volume and relevance metrics."""

    @classmethod
    def extract(
        cls,
        eval_item: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extracts retrieval metrics from evaluation pool item."""
        if not eval_item:
            return {
                "retrieved_evidence_count": 0,
                "relevant_evidence_count": 0,
                "supporting_retrieval_count": 0,
                "contradicting_retrieval_count": 0,
                "top_retrieval_score": -1.0,
                "retrieval_score_available": False,
            }

        pool = eval_item.get("candidate_evidence_pool", [])
        rel_ids = set(eval_item.get("relevant_evidence_ids", []))

        relevant_count = sum(1 for doc in pool if doc.get("is_relevant") or doc.get("evidence_id") in rel_ids)
        supporting_count = sum(1 for doc in pool if (doc.get("relation") == "SUPPORTS" or doc.get("evidence_relation") == "SUPPORTS"))
        contradicting_count = sum(1 for doc in pool if (doc.get("relation") == "CONTRADICTS" or doc.get("evidence_relation") == "CONTRADICTS"))

        # Candidate scores if available
        scores = [doc.get("candidate_similarity") for doc in pool if doc.get("candidate_similarity") is not None]
        top_score = max(scores) if scores else -1.0
        score_avail = len(scores) > 0

        return {
            "retrieved_evidence_count": len(pool),
            "relevant_evidence_count": relevant_count,
            "supporting_retrieval_count": supporting_count,
            "contradicting_retrieval_count": contradicting_count,
            "top_retrieval_score": round(top_score, 4),
            "retrieval_score_available": score_avail,
        }
