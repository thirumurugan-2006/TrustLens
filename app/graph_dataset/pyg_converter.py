"""
TrustLens PyTorch Geometric Converter (Phase 6C).
Converts HeterogeneousEvidenceGraph instances into torch_geometric.data.HeteroData.
Preserves node types, edge types, features, labels, masks, and split partitions.
"""

from typing import Any, Dict, List, Optional, Tuple

from app.graph_dataset.schemas import (
    EdgeType,
    GraphEdge,
    GraphNode,
    HeterogeneousEvidenceGraph,
    NodeType,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from torch_geometric.data import HeteroData
    PYG_AVAILABLE = True
except ImportError:
    HeteroData = None
    PYG_AVAILABLE = False


class PyGHeteroDataConverter:
    """
    Converts canonical HeterogeneousEvidenceGraph models into PyTorch Geometric HeteroData structures.
    Does not require PyG at runtime if only canonical JSON/JSONL serialization is used.
    """

    @staticmethod
    def is_available() -> bool:
        """Returns True if torch and torch_geometric are both importable."""
        return bool(TORCH_AVAILABLE and PYG_AVAILABLE)

    @classmethod
    def get_status(cls) -> str:
        """Reports PyG conversion availability status."""
        if cls.is_available():
            return "READY"
        if TORCH_AVAILABLE and not PYG_AVAILABLE:
            return "PYG_NOT_INSTALLED"
        return "TORCH_AND_PYG_NOT_INSTALLED"

    @classmethod
    def convert(
        cls,
        graph: HeterogeneousEvidenceGraph,
    ) -> Any:
        """
        Converts a HeterogeneousEvidenceGraph into a HeteroData object.
        Raises ImportError if torch_geometric is not available.
        """
        if not cls.is_available():
            raise ImportError(
                "torch_geometric is not installed in the environment. "
                "Canonical JSON/JSONL graph representations should be used."
            )

        data = HeteroData()

        # Group nodes by type
        nodes_by_type: Dict[str, List[GraphNode]] = {}
        node_id_to_idx: Dict[str, Tuple[str, int]] = {}

        for n in graph.nodes:
            tname = n.node_type.value.lower()
            idx = len(nodes_by_type.setdefault(tname, []))
            nodes_by_type[tname].append(n)
            node_id_to_idx[n.node_id] = (tname, idx)

        # Build node features
        for tname, nlist in nodes_by_type.items():
            features_list = []
            for n in nlist:
                fdict = n.features or {}
                # Numerical representation of categorical indexes
                fvec = [
                    float(fdict.get("language_idx", 0)),
                    float(fdict.get("platform_idx", 0)),
                    float(fdict.get("claim_type_idx", 0)),
                    float(fdict.get("relation_idx", 0)),
                    float(fdict.get("check_worthiness", 1.0)),
                    float(fdict.get("token_count", 0)),
                ]
                features_list.append(fvec)

            data[tname].x = torch.tensor(features_list, dtype=torch.float)
            data[tname].num_nodes = len(nlist)

        # Group edges by (src_type, edge_type, dst_type)
        edges_by_triplet: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}
        for e in graph.edges:
            src_info = node_id_to_idx.get(e.source_node_id)
            dst_info = node_id_to_idx.get(e.target_node_id)
            if not src_info or not dst_info:
                continue

            src_type, src_idx = src_info
            dst_type, dst_idx = dst_info
            rel_name = e.edge_type.value.lower()
            triplet = (src_type, rel_name, dst_type)
            edges_by_triplet.setdefault(triplet, []).append((src_idx, dst_idx))

        # Build edge_index tensors
        for triplet, edge_pairs in edges_by_triplet.items():
            src_indices = [p[0] for p in edge_pairs]
            dst_indices = [p[1] for p in edge_pairs]
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            data[triplet].edge_index = edge_index

        # Attach graph-level metadata and labels
        data.graph_id = graph.graph_id
        data.split = graph.split.value if hasattr(graph.split, "value") else str(graph.split)
        data.cluster_id = graph.cluster_id

        # Target post risk label if available
        post_labels = graph.labels.get("post_risk", {})
        if post_labels.get("has_risk_label"):
            data.y = torch.tensor([post_labels.get("risk_class_idx", -1)], dtype=torch.long)

        return data

    @classmethod
    def to_dict_representation(
        cls,
        graph: HeterogeneousEvidenceGraph,
    ) -> Dict[str, Any]:
        """
        PyG-equivalent dictionary structure for environments without torch_geometric installed.
        Allows offline validation of indices and features without external dependencies.
        """
        nodes_by_type: Dict[str, List[str]] = {}
        node_id_to_idx: Dict[str, Tuple[str, int]] = {}

        for n in graph.nodes:
            tname = n.node_type.value.lower()
            idx = len(nodes_by_type.setdefault(tname, []))
            nodes_by_type[tname].append(n.node_id)
            node_id_to_idx[n.node_id] = (tname, idx)

        edge_indices: Dict[str, List[List[int]]] = {}
        for e in graph.edges:
            src_info = node_id_to_idx.get(e.source_node_id)
            dst_info = node_id_to_idx.get(e.target_node_id)
            if not src_info or not dst_info:
                continue

            src_type, src_idx = src_info
            dst_type, dst_idx = dst_info
            rel_key = f"{src_type}__{e.edge_type.value.lower()}__{dst_type}"
            if rel_key not in edge_indices:
                edge_indices[rel_key] = [[], []]
            edge_indices[rel_key][0].append(src_idx)
            edge_indices[rel_key][1].append(dst_idx)

        return {
            "graph_id": graph.graph_id,
            "split": graph.split.value if hasattr(graph.split, "value") else str(graph.split),
            "cluster_id": graph.cluster_id,
            "node_types": {t: len(ids) for t, ids in nodes_by_type.items()},
            "edge_indices": edge_indices,
            "labels": graph.labels,
        }
