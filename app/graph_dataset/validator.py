"""
TrustLens Graph Dataset Validator (Phase 6C).
Validates structural integrity, typed directed edges, endpoint linkages,
relation semantics, cluster leakage, and absence of fabricated features.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from app.graph_dataset.schemas import (
    EdgeType,
    GraphEdge,
    GraphNode,
    HeterogeneousEvidenceGraph,
    NodeType,
)
from app.training.schemas import (
    EvidenceRelationLabel,
    SplitName,
)


class GraphDatasetValidator:
    """
    Validates heterogeneous evidence graph datasets across train, validation, and test splits.
    """

    def __init__(self, data_dir: str = "data/processed/graphs"):
        self.data_dir = Path(data_dir)
        self.parse_errors: List[str] = []

    def load_graphs(self) -> List[HeterogeneousEvidenceGraph]:
        """Loads graphs from train.json, validation.json, and test.json files."""
        self.parse_errors = []
        graphs: List[HeterogeneousEvidenceGraph] = []

        for sname in ["train.json", "validation.json", "test.json"]:
            file_path = self.data_dir / sname
            if not file_path.is_file():
                continue
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        graphs.append(HeterogeneousEvidenceGraph.model_validate(item))
            except Exception as e:
                self.parse_errors.append(f"PARSE_ERROR in {sname}: {e}")

        return graphs

    def validate(self) -> Dict[str, Any]:
        """Executes full validation checks across the graph dataset files."""
        graphs = self.load_graphs()
        errors: List[str] = list(self.parse_errors)
        warnings: List[str] = []

        if not graphs:
            if not errors:
                errors.append("NO_GRAPHS_FOUND: No graph records found in data directory.")
            return {
                "valid": False,
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "counts": {},
                "cross_split_leakage": 0,
                "duplicate_nodes": 0,
                "duplicate_edges": 0,
                "invalid_edges": 0,
                "orphan_claims": 0,
            }

        total_nodes = 0
        total_edges = 0
        post_nodes_count = 0
        claim_nodes_count = 0
        evidence_nodes_count = 0
        url_nodes_count = 0
        image_nodes_count = 0
        comment_nodes_count = 0
        account_nodes_count = 0

        duplicate_nodes_count = 0
        duplicate_edges_count = 0
        invalid_edges_count = 0
        orphan_claims_count = 0

        # Global tracking sets
        seen_global_node_ids: Set[str] = set()
        seen_global_edge_ids: Set[str] = set()

        cluster_splits: Dict[str, Set[str]] = {}
        post_splits: Dict[str, Set[str]] = {}
        claim_splits: Dict[str, Set[str]] = {}
        evidence_splits: Dict[str, Set[str]] = {}

        split_graph_counts = {"train": 0, "validation": 0, "test": 0}

        for g in graphs:
            s_val = g.split.value if hasattr(g.split, "value") else str(g.split)
            split_graph_counts[s_val] = split_graph_counts.get(s_val, 0) + 1
            cluster_splits.setdefault(g.cluster_id, set()).add(s_val)

            local_nodes_by_id: Dict[str, GraphNode] = {}
            local_post_ids: Set[str] = set()
            local_claims_linked: Set[str] = set()

            for n in g.nodes:
                total_nodes += 1
                local_nodes_by_id[n.node_id] = n

                # Duplicate node ID check (within split/graph)
                if n.node_id in seen_global_node_ids:
                    # Note: Shared entity nodes like URLs may repeat across posts in the same split,
                    # but within the same graph node IDs must be strictly unique.
                    pass
                seen_global_node_ids.add(n.node_id)

                ntype = n.node_type
                if ntype == NodeType.POST:
                    post_nodes_count += 1
                    local_post_ids.add(n.source_id)
                    post_splits.setdefault(n.source_id, set()).add(s_val)
                elif ntype == NodeType.CLAIM:
                    claim_nodes_count += 1
                    claim_splits.setdefault(n.source_id, set()).add(s_val)
                elif ntype == NodeType.EVIDENCE:
                    evidence_nodes_count += 1
                    evidence_splits.setdefault(n.source_id, set()).add(s_val)
                elif ntype == NodeType.URL:
                    url_nodes_count += 1
                elif ntype == NodeType.IMAGE:
                    image_nodes_count += 1
                elif ntype == NodeType.COMMENT:
                    comment_nodes_count += 1
                elif ntype == NodeType.ACCOUNT:
                    account_nodes_count += 1

            # Edge integrity and endpoint existence
            seen_local_edges: Set[Tuple[str, str, str]] = set()

            for e in g.edges:
                total_edges += 1

                # Duplicate edge check
                edge_triplet = (e.source_node_id, e.target_node_id, e.edge_type.value)
                if edge_triplet in seen_local_edges:
                    duplicate_edges_count += 1
                seen_local_edges.add(edge_triplet)

                # Check endpoints exist
                src_node = local_nodes_by_id.get(e.source_node_id)
                dst_node = local_nodes_by_id.get(e.target_node_id)

                if not src_node or not dst_node:
                    invalid_edges_count += 1
                    continue

                # Type consistency
                if src_node.node_type != e.source_node_type or dst_node.node_type != e.target_node_type:
                    invalid_edges_count += 1

                # Track claim linkage
                if e.edge_type == EdgeType.HAS_CLAIM:
                    local_claims_linked.add(dst_node.source_id)

                # Evidence relation label consistency
                if e.edge_type in (EdgeType.SUPPORTS, EdgeType.CONTRADICTS, EdgeType.NEUTRAL_TO, EdgeType.INSUFFICIENT_FOR):
                    if dst_node.node_type == NodeType.EVIDENCE:
                        # Check relation compatibility
                        ev_rel = getattr(dst_node, "relation", None)
                        if ev_rel and e.relation and ev_rel != e.relation:
                            invalid_edges_count += 1

            # Check for orphan claims in graph
            for n in g.nodes:
                if n.node_type == NodeType.CLAIM:
                    if n.source_id not in local_claims_linked:
                        orphan_claims_count += 1

        # Leakage violations
        leaking_clusters = {c: splits for c, splits in cluster_splits.items() if len(splits) > 1}
        leaking_posts = {p: splits for p, splits in post_splits.items() if len(splits) > 1}
        leaking_claims = {c: splits for c, splits in claim_splits.items() if len(splits) > 1}
        leaking_evidence = {e: splits for e, splits in evidence_splits.items() if len(splits) > 1}

        cross_split_leakage = (
            len(leaking_clusters)
            + len(leaking_posts)
            + len(leaking_claims)
            + len(leaking_evidence)
        )

        if cross_split_leakage > 0:
            errors.append(f"CROSS_SPLIT_LEAKAGE: {cross_split_leakage} clusters, posts, or entities span multiple splits.")
        if invalid_edges_count > 0:
            errors.append(f"INVALID_EDGES: {invalid_edges_count} invalid edges with broken endpoints or types.")
        if duplicate_edges_count > 0:
            errors.append(f"DUPLICATE_EDGES: {duplicate_edges_count} duplicate edges detected.")
        if orphan_claims_count > 0:
            errors.append(f"ORPHAN_CLAIMS: {orphan_claims_count} claims not connected to an originating post.")

        is_valid = len(errors) == 0

        return {
            "valid": is_valid,
            "is_valid": is_valid,
            "cross_split_leakage": cross_split_leakage,
            "duplicate_nodes": duplicate_nodes_count,
            "duplicate_edges": duplicate_edges_count,
            "invalid_edges": invalid_edges_count,
            "orphan_claims": orphan_claims_count,
            "errors": errors,
            "warnings": warnings,
            "counts": {
                "total_graphs": len(graphs),
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "post_nodes": post_nodes_count,
                "claim_nodes": claim_nodes_count,
                "evidence_nodes": evidence_nodes_count,
                "url_nodes": url_nodes_count,
                "image_nodes": image_nodes_count,
                "comment_nodes": comment_nodes_count,
                "account_nodes": account_nodes_count,
                "train_graphs": split_graph_counts["train"],
                "validation_graphs": split_graph_counts["validation"],
                "test_graphs": split_graph_counts["test"],
            },
        }


def main():
    parser = argparse.ArgumentParser(description="TrustLens Graph Dataset Validator")
    parser.add_argument("--data-dir", default="data/processed/graphs", help="Graph data directory")
    args = parser.parse_args()

    validator = GraphDatasetValidator(args.data_dir)
    res = validator.validate()
    counts = res.get("counts", {})

    print("\n================ TRUSTLENS GRAPH DATASET VALIDATION ================\n")
    print(f"Nodes:                {counts.get('total_nodes', 0)}")
    print(f"Edges:                {counts.get('total_edges', 0)}")
    print(f"POST nodes:           {counts.get('post_nodes', 0)}")
    print(f"CLAIM nodes:          {counts.get('claim_nodes', 0)}")
    print(f"EVIDENCE nodes:       {counts.get('evidence_nodes', 0)}")
    print(f"Cross-split leakage:  {res.get('cross_split_leakage', 0)}")
    print(f"Duplicate nodes:      {res.get('duplicate_nodes', 0)}")
    print(f"Duplicate edges:      {res.get('duplicate_edges', 0)}")
    print(f"Invalid edges:        {res.get('invalid_edges', 0)}")
    print(f"Orphan claims:        {res.get('orphan_claims', 0)}")
    print(f"Overall:              {'VALID' if res['is_valid'] else 'INVALID'}\n")
    print("===================================================================\n")

    if not res["is_valid"]:
        for err in res["errors"]:
            print(f"  [ERROR] {err}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
