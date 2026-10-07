"""
TrustLens Graph Serializer (Phase 6C).
Serializes HeterogeneousEvidenceGraph datasets into deterministic JSON and JSONL representations.
"""

import json
from pathlib import Path
from typing import Dict, List

from app.graph_dataset.schemas import (
    GraphEdge,
    GraphNode,
    HeterogeneousEvidenceGraph,
)


class GraphSerializer:
    """
    Handles file export of graphs, node tables, and edge tables for Train, Validation, and Test splits.
    """

    def __init__(self, output_dir: str = "data/processed/graphs"):
        self.output_dir = Path(output_dir)

    def serialize_split(
        self,
        split_name: str,
        graphs: List[HeterogeneousEvidenceGraph],
    ) -> Dict[str, str]:
        """
        Serializes graphs of a single split into:
        - {split_name}.json (full graph objects list)
        - {split_name}_nodes.jsonl (node catalog)
        - {split_name}_edges.jsonl (edge catalog)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # 1. Full Graphs JSON
        graphs_json_path = self.output_dir / f"{split_name}.json"
        graph_dicts = [g.model_dump() for g in graphs]
        with graphs_json_path.open("w", encoding="utf-8") as f:
            json.dump(graph_dicts, f, indent=2, ensure_ascii=False)
        paths["graphs"] = str(graphs_json_path)

        # 2. Nodes JSONL (deduplicated per split)
        nodes_jsonl_path = self.output_dir / f"{split_name}_nodes.jsonl"
        seen_nodes = set()
        with nodes_jsonl_path.open("w", encoding="utf-8") as f:
            for g in graphs:
                for n in g.nodes:
                    if n.node_id not in seen_nodes:
                        seen_nodes.add(n.node_id)
                        f.write(n.model_dump_json() + "\n")
        paths["nodes"] = str(nodes_jsonl_path)

        # 3. Edges JSONL (deduplicated per split)
        edges_jsonl_path = self.output_dir / f"{split_name}_edges.jsonl"
        seen_edges = set()
        with edges_jsonl_path.open("w", encoding="utf-8") as f:
            for g in graphs:
                for e in g.edges:
                    if e.edge_id not in seen_edges:
                        seen_edges.add(e.edge_id)
                        f.write(e.model_dump_json() + "\n")
        paths["edges"] = str(edges_jsonl_path)

        return paths

    def serialize_all(
        self,
        split_dict: Dict[str, List[HeterogeneousEvidenceGraph]],
    ) -> Dict[str, Dict[str, str]]:
        """Exports all splits (train, validation, test) to disk."""
        result = {}
        for sname, glist in split_dict.items():
            result[sname] = self.serialize_split(sname, glist)
        return result
