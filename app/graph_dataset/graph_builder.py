"""
TrustLens Evidence Graph Builder (Phase 6C).
Orchestrates nodes, edges, grounded features, and labels to construct HeterogeneousEvidenceGraph instances.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from app.graph_dataset.cluster_manager import GraphClusterManager
from app.graph_dataset.edge_builder import EdgeBuilder
from app.graph_dataset.feature_builder import GraphFeatureBuilder
from app.graph_dataset.label_builder import GraphLabelBuilder
from app.graph_dataset.node_builder import NodeBuilder
from app.graph_dataset.schemas import (
    AccountNode,
    ClaimNode,
    CommentNode,
    EdgeType,
    EvidenceNode,
    GraphEdge,
    GraphNode,
    HeterogeneousEvidenceGraph,
    ImageNode,
    NodeType,
    PostNode,
    UrlNode,
)
from app.training.schemas import (
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingRisk,
)


class EvidenceGraphBuilder:
    """
    Constructs discrete heterogeneous evidence graphs grounded in TrustLens posts, claims, and evidence.
    """

    def __init__(self):
        self.node_builder = NodeBuilder()
        self.edge_builder = EdgeBuilder()
        self.feature_builder = GraphFeatureBuilder()
        self.label_builder = GraphLabelBuilder()
        self.cluster_manager = GraphClusterManager()

    def build_graph_for_post(
        self,
        post: TrainingPost,
        claims: List[TrainingClaim],
        evidence_list: List[TrainingEvidence],
        risk_record: Optional[TrainingRisk] = None,
    ) -> HeterogeneousEvidenceGraph:
        """
        Constructs a complete HeterogeneousEvidenceGraph centered on a target post.
        """
        cluster_id = self.cluster_manager.get_cluster_id(post)
        split_val = (
            post.split_info.split
            if post.split_info and post.split_info.split
            else SplitName.train
        )
        if isinstance(split_val, str):
            split_val = SplitName(split_val)

        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        seen_node_ids: Set[str] = set()
        seen_edge_keys: Set[Tuple[str, str, str]] = set()

        def add_node(n: GraphNode):
            if n.node_id not in seen_node_ids:
                seen_node_ids.add(n.node_id)
                nodes.append(n)

        def add_edge(e: GraphEdge):
            key = (e.source_node_id, e.target_node_id, e.edge_type.value)
            if key not in seen_edge_keys:
                seen_edge_keys.add(key)
                edges.append(e)

        # 1. Post Node
        post_node = self.node_builder.build_post_node(post, cluster_id, split_val)
        add_node(post_node)

        # 2. Account Node & Edge
        account_node = self.node_builder.build_account_node(post, cluster_id, split_val)
        if account_node:
            add_node(account_node)
            assoc_edge = self.edge_builder.build_associated_with_edge(
                post_node, account_node, cluster_id, split_val
            )
            add_edge(assoc_edge)

        # 3. Media Image Nodes & Edges
        image_metas = getattr(post, "images_meta", []) or []
        for img_dict in image_metas:
            img_data = img_dict if isinstance(img_dict, dict) else img_dict.model_dump()
            img_node = self.node_builder.build_image_node(img_data, cluster_id, split_val)
            add_node(img_node)
            contains_edge = self.edge_builder.build_contains_image_edge(
                post_node, img_node, cluster_id, split_val
            )
            add_edge(contains_edge)

        # 4. Comments Nodes & Edges
        raw_comments = getattr(post, "comments", []) or []
        for cmt in raw_comments:
            cmt_data = cmt if isinstance(cmt, dict) else (cmt.model_dump() if hasattr(cmt, "model_dump") else {"text": str(cmt)})
            cmt_node = self.node_builder.build_comment_node(cmt_data, post.post_id, cluster_id, split_val)
            add_node(cmt_node)
            has_cmt_edge = self.edge_builder.build_has_comment_edge(
                post_node, cmt_node, cluster_id, split_val
            )
            add_edge(has_cmt_edge)

        # 5. Claim Nodes & Edges
        ev_by_claim: Dict[str, List[TrainingEvidence]] = {}
        for ev in evidence_list:
            ev_by_claim.setdefault(ev.claim_id, []).append(ev)

        all_evidence_nodes: List[EvidenceNode] = []

        for claim in claims:
            if claim.post_id != post.post_id:
                continue

            claim_node = self.node_builder.build_claim_node(claim, cluster_id, split_val)
            add_node(claim_node)

            has_claim_edge = self.edge_builder.build_has_claim_edge(
                post_node, claim_node, cluster_id, split_val
            )
            add_edge(has_claim_edge)

            # 6. Evidence Nodes & Edges
            claim_evs = ev_by_claim.get(claim.claim_id, [])
            for ev in claim_evs:
                ev_node = self.node_builder.build_evidence_node(ev, cluster_id, split_val)
                add_node(ev_node)
                all_evidence_nodes.append(ev_node)

                claim_ev_edge = self.edge_builder.build_claim_evidence_edge(
                    claim_node, ev_node, cluster_id, split_val, provenance=ev.provenance
                )
                add_edge(claim_ev_edge)

                # 7. URL Nodes & Edges (from evidence source_url)
                if ev.source_url and ev.source_url.strip():
                    url_node = self.node_builder.build_url_node(
                        ev.source_url, cluster_id, split_val
                    )
                    add_node(url_node)
                    sourced_edge = self.edge_builder.build_sourced_from_edge(
                        ev_node, url_node, cluster_id, split_val
                    )
                    add_edge(sourced_edge)

        # Attach Grounded Features
        for n in nodes:
            self.feature_builder.attach_features_to_node(
                n,
                claim_count=len(claims),
                image_count=len(image_metas),
                comment_count=len(raw_comments),
            )

        # Attach Grounded Labels
        post_labels = self.label_builder.build_post_labels(post_node, risk_record)
        claim_labels = self.label_builder.build_claim_relation_labels(all_evidence_nodes)
        combined_labels = {
            "post_risk": post_labels,
            "claim_relations": claim_labels,
            "gat_task_status": GraphLabelBuilder.GAT_TASK_STATUS,
        }

        graph_id = f"graph_{post.post_id}"
        return HeterogeneousEvidenceGraph(
            graph_id=graph_id,
            root_post_id=post.post_id,
            split=split_val,
            cluster_id=cluster_id,
            nodes=nodes,
            edges=edges,
            labels=combined_labels,
            metadata={
                "language": post_node.language,
                "platform": post_node.platform,
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        )
