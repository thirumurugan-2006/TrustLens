"""
TrustLens Graph Cluster Manager (Phase 6C).
Manages leakage grouping across campaigns, translation groups, and source accounts.
Guarantees that related posts, claims, evidence, and entity nodes move as indivisible graph blocks.
"""

from typing import Any, Dict, List, Set, Union

from app.graph_dataset.schemas import (
    GraphEdge,
    GraphNode,
    HeterogeneousEvidenceGraph,
)
from app.training.schemas import (
    SplitMetadata,
    TrainingClaim,
    TrainingPost,
)


class GraphClusterManager:
    """
    Derives canonical cluster identifiers and audits cross-split leakage for heterogeneous evidence graphs.
    """

    @classmethod
    def get_cluster_id(cls, item: Any) -> str:
        """Derives canonical cluster identifier from post, claim, or split metadata."""
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
        if hasattr(item, "cluster_id"):
            return str(item.cluster_id)
        return f"generic_cluster_{id(item)}"

    @staticmethod
    def audit_leakage(
        graphs: List[HeterogeneousEvidenceGraph],
    ) -> Dict[str, Any]:
        """
        Audits cross-split leakage across cluster IDs, post IDs, claim IDs, and evidence IDs.
        Returns leakage audit dictionary with violation count and details.
        """
        cluster_splits: Dict[str, Set[str]] = {}
        post_splits: Dict[str, Set[str]] = {}
        claim_splits: Dict[str, Set[str]] = {}
        evidence_splits: Dict[str, Set[str]] = {}

        for g in graphs:
            s_val = g.split.value if hasattr(g.split, "value") else str(g.split)
            cluster_splits.setdefault(g.cluster_id, set()).add(s_val)

            for n in g.nodes:
                if n.node_type.value == "POST":
                    post_splits.setdefault(n.source_id, set()).add(s_val)
                elif n.node_type.value == "CLAIM":
                    claim_splits.setdefault(n.source_id, set()).add(s_val)
                elif n.node_type.value == "EVIDENCE":
                    evidence_splits.setdefault(n.source_id, set()).add(s_val)

        leaking_clusters = {c: s for c, s in cluster_splits.items() if len(s) > 1}
        leaking_posts = {p: s for p, s in post_splits.items() if len(s) > 1}
        leaking_claims = {c: s for c, s in claim_splits.items() if len(s) > 1}
        leaking_evidence = {e: s for e, s in evidence_splits.items() if len(s) > 1}

        total_violations = (
            len(leaking_clusters)
            + len(leaking_posts)
            + len(leaking_claims)
            + len(leaking_evidence)
        )

        return {
            "cross_split_leakage_violations": total_violations,
            "leaking_clusters_count": len(leaking_clusters),
            "leaking_posts_count": len(leaking_posts),
            "leaking_claims_count": len(leaking_claims),
            "leaking_evidence_count": len(leaking_evidence),
            "leaking_clusters": leaking_clusters,
            "leakage_free": total_violations == 0,
        }
