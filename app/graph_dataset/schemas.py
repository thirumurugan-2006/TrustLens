"""
TrustLens Heterogeneous Evidence Graph Dataset Schemas (Phase 6C).
Typed directed heterogeneous graph representing posts, claims, evidence,
URLs, images, comments, and accounts with full provenance and cluster isolation.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

from app.training.schemas import (
    ClaimType,
    EvidenceRelationLabel,
    LanguageMetadata,
    ProvenanceMetadata,
    ProvenanceSourceType,
    RiskLevel,
    SplitName,
)


class NodeType(str, Enum):
    """Controlled taxonomy of heterogeneous node types."""
    POST = "POST"
    CLAIM = "CLAIM"
    EVIDENCE = "EVIDENCE"
    URL = "URL"
    IMAGE = "IMAGE"
    COMMENT = "COMMENT"
    ACCOUNT = "ACCOUNT"


class EdgeType(str, Enum):
    """Controlled taxonomy of typed directed heterogeneous edge relationships."""
    HAS_CLAIM = "HAS_CLAIM"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL_TO = "NEUTRAL_TO"
    INSUFFICIENT_FOR = "INSUFFICIENT_FOR"
    SOURCED_FROM = "SOURCED_FROM"
    CONTAINS = "CONTAINS"
    HAS_COMMENT = "HAS_COMMENT"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    SIMILAR_TO = "SIMILAR_TO"


# ---------------------------------------------------------------------------
# Node Schemas
# ---------------------------------------------------------------------------

class GraphNode(BaseModel):
    """
    Base heterogeneous graph node.
    Every node must have node_id, node_type, source_id, split, cluster_id, and metadata.
    """
    node_id: str
    node_type: NodeType
    source_id: str
    split: SplitName = SplitName.train
    cluster_id: str
    features: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PostNode(GraphNode):
    """Post entity node."""
    node_type: NodeType = NodeType.POST
    post_id: str
    text: str
    language: str = "en"
    script: str = "Latin"
    platform: str = "unknown"

    @model_validator(mode="before")
    @classmethod
    def sync_post_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "post_id" in data and "source_id" not in data:
                data["source_id"] = data["post_id"]
        return data


class ClaimNode(GraphNode):
    """Claim entity node."""
    node_type: NodeType = NodeType.CLAIM
    claim_id: str
    parent_post_id: str
    atomic_claim_id: Optional[str] = None
    claim_text: str
    claim_type: str = "UNCLASSIFIED"
    language: str = "en"

    @model_validator(mode="before")
    @classmethod
    def sync_claim_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "claim_id" in data and "source_id" not in data:
                data["source_id"] = data["claim_id"]
        return data


class EvidenceNode(GraphNode):
    """Evidence document entity node."""
    node_type: NodeType = NodeType.EVIDENCE
    evidence_id: str
    claim_id: str
    evidence_text: str
    source_type: str = "REGULATORY_ADVISORY"
    source_url: Optional[str] = None
    relation: EvidenceRelationLabel = EvidenceRelationLabel.CONTRADICTS
    provenance: Optional[ProvenanceMetadata] = None

    @model_validator(mode="before")
    @classmethod
    def sync_evidence_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "evidence_id" in data and "source_id" not in data:
                data["source_id"] = data["evidence_id"]
        return data


class UrlNode(GraphNode):
    """Web URL entity node."""
    node_type: NodeType = NodeType.URL
    url_id: str
    url: str
    domain: str

    @model_validator(mode="before")
    @classmethod
    def sync_url_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "url_id" in data and "source_id" not in data:
                data["source_id"] = data["url_id"]
        return data


class ImageNode(GraphNode):
    """Media image entity node."""
    node_type: NodeType = NodeType.IMAGE
    image_id: str
    image_hash: Optional[str] = None
    phash: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def sync_image_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "image_id" in data and "source_id" not in data:
                data["source_id"] = data["image_id"]
        return data


class CommentNode(GraphNode):
    """Social commentary entity node."""
    node_type: NodeType = NodeType.COMMENT
    comment_id: str
    parent_post_id: str
    comment_text: str
    stance: Optional[str] = None
    language: str = "en"

    @model_validator(mode="before")
    @classmethod
    def sync_comment_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "comment_id" in data and "source_id" not in data:
                data["source_id"] = data["comment_id"]
        return data


class AccountNode(GraphNode):
    """Author or social account entity node."""
    node_type: NodeType = NodeType.ACCOUNT
    account_id: str
    platform: str = "unknown"
    username: Optional[str] = None
    is_verified: bool = False

    @model_validator(mode="before")
    @classmethod
    def sync_account_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "account_id" in data and "source_id" not in data:
                data["source_id"] = data["account_id"]
        return data


# ---------------------------------------------------------------------------
# Edge Schema
# ---------------------------------------------------------------------------

class GraphEdge(BaseModel):
    """
    Typed directed heterogeneous graph edge.
    Every edge contains edge_id, source/target nodes and types, edge_type, split, cluster_id,
    provenance, and optional evidence relation.
    """
    edge_id: str
    source_node_id: str
    source_node_type: NodeType
    target_node_id: str
    target_node_type: NodeType
    edge_type: EdgeType
    relation: Optional[EvidenceRelationLabel] = None
    split: SplitName = SplitName.train
    cluster_id: str
    provenance: Optional[ProvenanceMetadata] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_endpoints(self):
        # Prevent self loops
        if self.source_node_id == self.target_node_id:
            raise ValueError(f"Self-loop forbidden: source {self.source_node_id} equals target.")
        return self


# ---------------------------------------------------------------------------
# Graph Container Schema
# ---------------------------------------------------------------------------

class HeterogeneousEvidenceGraph(BaseModel):
    """
    Discrete heterogeneous evidence graph (e.g., per-post or per-campaign subgraph).
    """
    graph_id: str
    root_post_id: Optional[str] = None
    split: SplitName = SplitName.train
    cluster_id: str
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    labels: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)
