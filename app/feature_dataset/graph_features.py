"""
TrustLens Graph Feature Extractor (Phase 6D).
Derives grounded graph structural statistics from Phase 6C heterogeneous evidence graphs.
Strict rule: Never use GAT embeddings or fabricated graph representations.
"""

from typing import Any, Dict, Optional

from app.graph_dataset.schemas import HeterogeneousEvidenceGraph


class GraphFeatureExtractor:
    """Extracts structural topological statistics from HeterogeneousEvidenceGraph."""

    @classmethod
    def extract(
        cls,
        graph: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Extracts graph counts and degree statistics from HeterogeneousEvidenceGraph or dict."""
        if not graph:
            return {
                "graph_node_count": 0,
                "graph_edge_count": 0,
                "graph_claim_nodes": 0,
                "graph_evidence_nodes": 0,
                "graph_url_nodes": 0,
                "graph_account_nodes": 0,
                "graph_support_edges": 0,
                "graph_contradict_edges": 0,
                "graph_neutral_edges": 0,
                "graph_insufficient_edges": 0,
                "graph_degree_mean": 0.0,
            }

        nodes = getattr(graph, "nodes", None) if hasattr(graph, "nodes") else graph.get("nodes", [])
        edges = getattr(graph, "edges", None) if hasattr(graph, "edges") else graph.get("edges", [])

        node_count = len(nodes)
        edge_count = len(edges)

        def get_type_str(item: Any, key: str) -> str:
            val = getattr(item, key, None) if hasattr(item, key) else (item.get(key) if isinstance(item, dict) else None)
            return str(getattr(val, "value", val) or "").upper()

        claim_nodes = sum(1 for n in nodes if get_type_str(n, "node_type") == "CLAIM")
        evidence_nodes = sum(1 for n in nodes if get_type_str(n, "node_type") == "EVIDENCE")
        url_nodes = sum(1 for n in nodes if get_type_str(n, "node_type") == "URL")
        account_nodes = sum(1 for n in nodes if get_type_str(n, "node_type") == "ACCOUNT")

        support_edges = sum(1 for e in edges if get_type_str(e, "edge_type") == "SUPPORTS")
        contradict_edges = sum(1 for e in edges if get_type_str(e, "edge_type") == "CONTRADICTS")
        neutral_edges = sum(1 for e in edges if get_type_str(e, "edge_type") in ("NEUTRAL_TO", "NEUTRAL"))
        insufficient_edges = sum(1 for e in edges if get_type_str(e, "edge_type") in ("INSUFFICIENT_FOR", "INSUFFICIENT"))

        degree_mean = round((2.0 * edge_count) / max(node_count, 1), 4)

        return {
            "graph_node_count": node_count,
            "graph_edge_count": edge_count,
            "graph_claim_nodes": claim_nodes,
            "graph_evidence_nodes": evidence_nodes,
            "graph_url_nodes": url_nodes,
            "graph_account_nodes": account_nodes,
            "graph_support_edges": support_edges,
            "graph_contradict_edges": contradict_edges,
            "graph_neutral_edges": neutral_edges,
            "graph_insufficient_edges": insufficient_edges,
            "graph_degree_mean": degree_mean,
        }
