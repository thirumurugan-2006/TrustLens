"""
TrustLens Heterogeneous Evidence Graph Dataset Module (Phase 6C).
Constructs, manages, validates, and serializes typed directed heterogeneous evidence graphs
for future Graph Attention Network (GAT) verification without training any neural models.
"""

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from app.graph_dataset.node_builder import NodeBuilder
    from app.graph_dataset.edge_builder import EdgeBuilder
    from app.graph_dataset.feature_builder import GraphFeatureBuilder
    from app.graph_dataset.label_builder import GraphLabelBuilder
    from app.graph_dataset.cluster_manager import GraphClusterManager
    from app.graph_dataset.splitter import GraphSplitter
    from app.graph_dataset.graph_builder import EvidenceGraphBuilder
    from app.graph_dataset.serializer import GraphSerializer
    from app.graph_dataset.pyg_converter import PyGHeteroDataConverter
    from app.graph_dataset.validator import GraphDatasetValidator
    from app.graph_dataset.manifest import GraphDatasetManifest, GraphManifestBuilder
    from app.graph_dataset.builder import GraphDatasetPipeline

__all__ = [
    "NodeType",
    "EdgeType",
    "GraphNode",
    "PostNode",
    "ClaimNode",
    "EvidenceNode",
    "UrlNode",
    "ImageNode",
    "CommentNode",
    "AccountNode",
    "GraphEdge",
    "HeterogeneousEvidenceGraph",
    "NodeBuilder",
    "EdgeBuilder",
    "GraphFeatureBuilder",
    "GraphLabelBuilder",
    "GraphClusterManager",
    "GraphSplitter",
    "EvidenceGraphBuilder",
    "GraphSerializer",
    "PyGHeteroDataConverter",
    "GraphDatasetValidator",
    "GraphDatasetManifest",
    "GraphManifestBuilder",
    "GraphDatasetPipeline",
]


def __getattr__(name: str):
    if name == "NodeBuilder":
        from app.graph_dataset.node_builder import NodeBuilder
        return NodeBuilder
    elif name == "EdgeBuilder":
        from app.graph_dataset.edge_builder import EdgeBuilder
        return EdgeBuilder
    elif name == "GraphFeatureBuilder":
        from app.graph_dataset.feature_builder import GraphFeatureBuilder
        return GraphFeatureBuilder
    elif name == "GraphLabelBuilder":
        from app.graph_dataset.label_builder import GraphLabelBuilder
        return GraphLabelBuilder
    elif name == "GraphClusterManager":
        from app.graph_dataset.cluster_manager import GraphClusterManager
        return GraphClusterManager
    elif name == "GraphSplitter":
        from app.graph_dataset.splitter import GraphSplitter
        return GraphSplitter
    elif name == "EvidenceGraphBuilder":
        from app.graph_dataset.graph_builder import EvidenceGraphBuilder
        return EvidenceGraphBuilder
    elif name == "GraphSerializer":
        from app.graph_dataset.serializer import GraphSerializer
        return GraphSerializer
    elif name == "PyGHeteroDataConverter":
        from app.graph_dataset.pyg_converter import PyGHeteroDataConverter
        return PyGHeteroDataConverter
    elif name == "GraphDatasetValidator":
        from app.graph_dataset.validator import GraphDatasetValidator
        return GraphDatasetValidator
    elif name == "GraphDatasetManifest":
        from app.graph_dataset.manifest import GraphDatasetManifest
        return GraphDatasetManifest
    elif name == "GraphManifestBuilder":
        from app.graph_dataset.manifest import GraphManifestBuilder
        return GraphManifestBuilder
    elif name == "GraphDatasetPipeline":
        from app.graph_dataset.builder import GraphDatasetPipeline
        return GraphDatasetPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
