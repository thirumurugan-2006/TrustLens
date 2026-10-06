from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class GraphNode(BaseModel):
    node_id: str
    node_type: str
    attributes: Dict[str, Any]

class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: str

class EvidenceGraph(BaseModel):
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
