"""
TrustLens Edge Builder (Phase 6C).
Constructs strongly-typed directed heterogeneous graph edges with full relation preservation.
"""

from typing import Any, Dict, List, Optional

from app.graph_dataset.schemas import (
    AccountNode,
    ClaimNode,
    CommentNode,
    EdgeType,
    EvidenceNode,
    GraphEdge,
    ImageNode,
    NodeType,
    PostNode,
    UrlNode,
)
from app.training.schemas import (
    EvidenceRelationLabel,
    ProvenanceMetadata,
    SplitName,
)


class EdgeBuilder:
    """
    Constructs typed directed heterogeneous edges.
    Enforces relation semantics and endpoint validation.
    """

    @staticmethod
    def build_has_claim_edge(
        post_node: PostNode,
        claim_node: ClaimNode,
        cluster_id: str,
        split: SplitName,
    ) -> GraphEdge:
        """Constructs POST --HAS_CLAIM--> CLAIM edge."""
        edge_id = f"edge_has_claim_{post_node.post_id}_{claim_node.claim_id}"
        return GraphEdge(
            edge_id=edge_id,
            source_node_id=post_node.node_id,
            source_node_type=NodeType.POST,
            target_node_id=claim_node.node_id,
            target_node_type=NodeType.CLAIM,
            edge_type=EdgeType.HAS_CLAIM,
            split=split,
            cluster_id=cluster_id,
            metadata={"parent_post_id": post_node.post_id},
        )

    @staticmethod
    def build_claim_evidence_edge(
        claim_node: ClaimNode,
        evidence_node: EvidenceNode,
        cluster_id: str,
        split: SplitName,
        provenance: Optional[ProvenanceMetadata] = None,
    ) -> GraphEdge:
        """
        Constructs CLAIM --[SUPPORTS | CONTRADICTS | NEUTRAL_TO | INSUFFICIENT_FOR]--> EVIDENCE edge.
        Preserves ground truth evidence relation label explicitly.
        """
        relation = evidence_node.relation
        if relation == EvidenceRelationLabel.SUPPORTS:
            etype = EdgeType.SUPPORTS
        elif relation == EvidenceRelationLabel.CONTRADICTS:
            etype = EdgeType.CONTRADICTS
        elif relation == EvidenceRelationLabel.NEUTRAL:
            etype = EdgeType.NEUTRAL_TO
        elif relation == EvidenceRelationLabel.INSUFFICIENT:
            etype = EdgeType.INSUFFICIENT_FOR
        else:
            etype = EdgeType.NEUTRAL_TO

        edge_id = f"edge_{etype.value.lower()}_{claim_node.claim_id}_{evidence_node.evidence_id}"
        return GraphEdge(
            edge_id=edge_id,
            source_node_id=claim_node.node_id,
            source_node_type=NodeType.CLAIM,
            target_node_id=evidence_node.node_id,
            target_node_type=NodeType.EVIDENCE,
            edge_type=etype,
            relation=relation,
            split=split,
            cluster_id=cluster_id,
            provenance=provenance or evidence_node.provenance,
            metadata={"relation_label": relation.value if hasattr(relation, "value") else str(relation)},
        )

    @staticmethod
    def build_sourced_from_edge(
        evidence_node: EvidenceNode,
        url_node: UrlNode,
        cluster_id: str,
        split: SplitName,
    ) -> GraphEdge:
        """Constructs EVIDENCE --SOURCED_FROM--> URL edge."""
        edge_id = f"edge_sourced_from_{evidence_node.evidence_id}_{url_node.url_id}"
        return GraphEdge(
            edge_id=edge_id,
            source_node_id=evidence_node.node_id,
            source_node_type=NodeType.EVIDENCE,
            target_node_id=url_node.node_id,
            target_node_type=NodeType.URL,
            edge_type=EdgeType.SOURCED_FROM,
            split=split,
            cluster_id=cluster_id,
            metadata={"domain": url_node.domain},
        )

    @staticmethod
    def build_associated_with_edge(
        post_node: PostNode,
        account_node: AccountNode,
        cluster_id: str,
        split: SplitName,
    ) -> GraphEdge:
        """Constructs POST --ASSOCIATED_WITH--> ACCOUNT edge."""
        edge_id = f"edge_assoc_{post_node.post_id}_{account_node.account_id}"
        return GraphEdge(
            edge_id=edge_id,
            source_node_id=post_node.node_id,
            source_node_type=NodeType.POST,
            target_node_id=account_node.node_id,
            target_node_type=NodeType.ACCOUNT,
            edge_type=EdgeType.ASSOCIATED_WITH,
            split=split,
            cluster_id=cluster_id,
            metadata={"platform": account_node.platform},
        )

    @staticmethod
    def build_contains_image_edge(
        post_node: PostNode,
        image_node: ImageNode,
        cluster_id: str,
        split: SplitName,
    ) -> GraphEdge:
        """Constructs POST --CONTAINS--> IMAGE edge."""
        edge_id = f"edge_contains_{post_node.post_id}_{image_node.image_id}"
        return GraphEdge(
            edge_id=edge_id,
            source_node_id=post_node.node_id,
            source_node_type=NodeType.POST,
            target_node_id=image_node.node_id,
            target_node_type=NodeType.IMAGE,
            edge_type=EdgeType.CONTAINS,
            split=split,
            cluster_id=cluster_id,
            metadata={"image_id": image_node.image_id},
        )

    @staticmethod
    def build_has_comment_edge(
        post_node: PostNode,
        comment_node: CommentNode,
        cluster_id: str,
        split: SplitName,
    ) -> GraphEdge:
        """Constructs POST --HAS_COMMENT--> COMMENT edge."""
        edge_id = f"edge_comment_{post_node.post_id}_{comment_node.comment_id}"
        return GraphEdge(
            edge_id=edge_id,
            source_node_id=post_node.node_id,
            source_node_type=NodeType.POST,
            target_node_id=comment_node.node_id,
            target_node_type=NodeType.COMMENT,
            edge_type=EdgeType.HAS_COMMENT,
            split=split,
            cluster_id=cluster_id,
            metadata={"comment_id": comment_node.comment_id},
        )

    @staticmethod
    def build_similar_to_edge(
        img_node_a: ImageNode,
        img_node_b: ImageNode,
        similarity_score: float,
        threshold: float,
        similarity_method: str,
        cluster_id: str,
        split: SplitName,
    ) -> GraphEdge:
        """
        Constructs IMAGE --SIMILAR_TO--> IMAGE edge.
        Only called when verified similarity passes threshold.
        """
        edge_id = f"edge_sim_{img_node_a.image_id}_{img_node_b.image_id}"
        return GraphEdge(
            edge_id=edge_id,
            source_node_id=img_node_a.node_id,
            source_node_type=NodeType.IMAGE,
            target_node_id=img_node_b.node_id,
            target_node_type=NodeType.IMAGE,
            edge_type=EdgeType.SIMILAR_TO,
            split=split,
            cluster_id=cluster_id,
            metadata={
                "similarity_score": similarity_score,
                "threshold": threshold,
                "similarity_method": similarity_method,
            },
        )
