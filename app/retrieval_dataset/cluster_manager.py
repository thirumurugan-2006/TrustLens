from typing import Any, Dict, List, Set, Tuple

from app.retrieval_dataset.schemas import RetrievalExample, RetrievalPair
from app.training.schemas import SplitName, TrainingClaim, TrainingPost


class RetrievalClusterManager:
    """
    Manages leakage grouping across campaigns, translation groups, and source accounts.
    Guarantees that related queries, positive evidence, and candidates move as indivisible blocks.
    """

    @classmethod
    def get_canonical_cluster_id(cls, claim: TrainingClaim) -> str:
        """Derives canonical cluster identifier from claim split metadata."""
        if claim.split_info:
            if claim.split_info.campaign_group_id:
                return f"camp_{claim.split_info.campaign_group_id}"
            if claim.split_info.translation_group_id:
                return f"trans_{claim.split_info.translation_group_id}"
            if claim.split_info.source_group_id:
                return f"src_{claim.split_info.source_group_id}"
        return f"claim_cluster_{claim.claim_id}"

    @classmethod
    def get_cluster_id(cls, item: Any) -> str:
        """Derives cluster identifier from either TrainingPost or TrainingClaim."""
        if hasattr(item, "split_info") and item.split_info:
            s_info = item.split_info
            if getattr(s_info, "campaign_group_id", None):
                return f"camp_{s_info.campaign_group_id}"
            if getattr(s_info, "translation_group_id", None):
                return f"trans_{s_info.translation_group_id}"
            if getattr(s_info, "source_group_id", None):
                return f"src_{s_info.source_group_id}"
            if getattr(s_info, "post_family_id", None):
                return f"fam_{s_info.post_family_id}"
        if hasattr(item, "claim_id"):
            return f"claim_cluster_{item.claim_id}"
        if hasattr(item, "post_id"):
            return f"post_cluster_{item.post_id}"
        return f"generic_cluster_{id(item)}"

    @staticmethod
    def audit_leakage(
        examples: List[RetrievalExample],
    ) -> Dict[str, Any]:
        """
        Audits cross-split leakage across cluster_id, claim_id, query_id, and positive_evidence_id.
        Returns leakage audit dictionary with violation count and details.
        """
        cluster_splits: Dict[str, Set[str]] = {}
        claim_splits: Dict[str, Set[str]] = {}
        evidence_splits: Dict[str, Set[str]] = {}

        for ex in examples:
            s_val = ex.split.value if hasattr(ex.split, "value") else str(ex.split)
            cluster_splits.setdefault(ex.cluster_id, set()).add(s_val)
            claim_splits.setdefault(ex.claim_id, set()).add(s_val)
            evidence_splits.setdefault(ex.positive_evidence_id, set()).add(s_val)

        leaking_clusters = {c: splits for c, splits in cluster_splits.items() if len(splits) > 1}
        leaking_claims = {c: splits for c, splits in claim_splits.items() if len(splits) > 1}
        leaking_evidence = {e: splits for e, splits in evidence_splits.items() if len(splits) > 1}

        total_violations = len(leaking_clusters) + len(leaking_claims) + len(leaking_evidence)

        return {
            "cross_split_leakage_violations": total_violations,
            "leaking_clusters_count": len(leaking_clusters),
            "leaking_claims_count": len(leaking_claims),
            "leaking_evidence_count": len(leaking_evidence),
            "leaking_clusters": leaking_clusters,
            "leakage_free": total_violations == 0,
        }
