"""
TrustLens Graph Dataset Manifest Builder (Phase 6C).
Compiles verified metadata, node/edge distributions, and graph topology statistics.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.graph_dataset.pyg_converter import PyGHeteroDataConverter
from app.graph_dataset.schemas import (
    EdgeType,
    GraphEdge,
    GraphNode,
    HeterogeneousEvidenceGraph,
    NodeType,
)


class GraphDatasetManifest(BaseModel):
    """Manifest specification for TrustLens Heterogeneous Evidence Graph Dataset."""
    dataset_id: str = "trustlens_graph_v0.1.0"
    graph_dataset_version: str = "v0.1.0"
    source_dataset_version: str = "v0.2.0"
    retrieval_dataset_version: str = "v0.1.0"
    schema_version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    graph_count: int
    node_count: int
    edge_count: int
    density: float

    node_type_distribution: Dict[str, int]
    edge_type_distribution: Dict[str, int]
    relation_distribution: Dict[str, int]
    risk_distribution: Dict[str, int]
    language_distribution: Dict[str, int]

    train_graphs: int
    validation_graphs: int
    test_graphs: int
    cluster_count: int

    average_nodes_per_graph: float
    average_edges_per_graph: float
    average_claims_per_post: float
    average_evidence_per_claim: float

    cross_split_leakage: int = 0
    duplicate_nodes: int = 0
    duplicate_edges: int = 0
    invalid_edges: int = 0
    orphan_claims: int = 0
    pyg_conversion_status: str = "READY"


class GraphManifestBuilder:
    """Builds GraphDatasetManifest from validated HeterogeneousEvidenceGraph collections."""

    @staticmethod
    def build_manifest(
        graphs: List[HeterogeneousEvidenceGraph],
        validation_report: Optional[Dict[str, Any]] = None,
    ) -> GraphDatasetManifest:
        """Computes all summary statistics and distributions from graphs."""
        node_type_dist: Dict[str, int] = {}
        edge_type_dist: Dict[str, int] = {}
        relation_dist: Dict[str, int] = {}
        risk_dist: Dict[str, int] = {}
        lang_dist: Dict[str, int] = {}

        total_nodes = 0
        total_edges = 0
        total_claims = 0
        total_evidence = 0
        total_posts = 0

        clusters = set()
        splits_count = {"train": 0, "validation": 0, "test": 0}

        for g in graphs:
            s_name = g.split.value if hasattr(g.split, "value") else str(g.split)
            splits_count[s_name] = splits_count.get(s_name, 0) + 1
            clusters.add(g.cluster_id)

            # Node stats
            for n in g.nodes:
                total_nodes += 1
                tname = n.node_type.value
                node_type_dist[tname] = node_type_dist.get(tname, 0) + 1

                if tname == "POST":
                    total_posts += 1
                    lang = getattr(n, "language", "en")
                    lang_dist[lang] = lang_dist.get(lang, 0) + 1
                elif tname == "CLAIM":
                    total_claims += 1
                elif tname == "EVIDENCE":
                    total_evidence += 1

            # Edge stats
            for e in g.edges:
                total_edges += 1
                ename = e.edge_type.value
                edge_type_dist[ename] = edge_type_dist.get(ename, 0) + 1

                if e.relation:
                    rel_name = e.relation.value if hasattr(e.relation, "value") else str(e.relation)
                    relation_dist[rel_name] = relation_dist.get(rel_name, 0) + 1

            # Risk label stats
            post_labels = g.labels.get("post_risk", {})
            rlevel = post_labels.get("risk_level")
            if rlevel and rlevel != "UNKNOWN":
                risk_dist[rlevel] = risk_dist.get(rlevel, 0) + 1

        num_graphs = len(graphs) if graphs else 1
        avg_nodes = round(total_nodes / num_graphs, 2)
        avg_edges = round(total_edges / num_graphs, 2)
        avg_claims = round(total_claims / (total_posts or 1), 2)
        avg_ev = round(total_evidence / (total_claims or 1), 2)

        # Graph density: E / (V * (V - 1)) summed or average
        possible_edges = total_nodes * (total_nodes - 1) if total_nodes > 1 else 1
        density = round(total_edges / possible_edges, 6) if possible_edges > 0 else 0.0

        val = validation_report or {}

        return GraphDatasetManifest(
            graph_count=len(graphs),
            node_count=total_nodes,
            edge_count=total_edges,
            density=density,
            node_type_distribution=node_type_dist,
            edge_type_distribution=edge_type_dist,
            relation_distribution=relation_dist,
            risk_distribution=risk_dist,
            language_distribution=lang_dist,
            train_graphs=splits_count.get("train", 0),
            validation_graphs=splits_count.get("validation", 0),
            test_graphs=splits_count.get("test", 0),
            cluster_count=len(clusters),
            average_nodes_per_graph=avg_nodes,
            average_edges_per_graph=avg_edges,
            average_claims_per_post=avg_claims,
            average_evidence_per_claim=avg_ev,
            cross_split_leakage=val.get("cross_split_leakage", 0),
            duplicate_nodes=val.get("duplicate_nodes", 0),
            duplicate_edges=val.get("duplicate_edges", 0),
            invalid_edges=val.get("invalid_edges", 0),
            orphan_claims=val.get("orphan_claims", 0),
            pyg_conversion_status=PyGHeteroDataConverter.get_status(),
        )
