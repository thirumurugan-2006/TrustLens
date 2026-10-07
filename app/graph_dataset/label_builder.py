"""
TrustLens Graph Label Builder (Phase 6C).
Prepares grounded multi-task evaluation and training labels for future GAT experiments.
Preserves clear scientific boundary between claim relation labels and post risk labels.
"""

from typing import Any, Dict, List, Optional

from app.graph_dataset.schemas import (
    ClaimNode,
    EvidenceNode,
    PostNode,
)
from app.training.schemas import (
    EvidenceRelationLabel,
    RiskLevel,
    TrainingRisk,
)


class GraphLabelBuilder:
    """
    Constructs graph target dictionaries for future GAT architectures.
    Strictly documents GAT_TARGET_PENDING and never trains any neural model.
    """

    GAT_TASK_STATUS = "GAT_TARGET_PENDING"

    RISK_MAP = {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2,
        "INSUFFICIENT": 3,
    }

    RELATION_MAP = {
        "SUPPORTS": 0,
        "CONTRADICTS": 1,
        "NEUTRAL": 2,
        "INSUFFICIENT": 3,
    }

    @classmethod
    def build_post_labels(
        cls,
        post_node: PostNode,
        risk_record: Optional[TrainingRisk] = None,
    ) -> Dict[str, Any]:
        """Builds post-level risk labels from ground truth TrainingRisk record."""
        if not risk_record:
            return {
                "risk_level": "UNKNOWN",
                "risk_class_idx": -1,
                "has_risk_label": False,
                "gat_target_status": cls.GAT_TASK_STATUS,
            }

        risk_str = (
            risk_record.risk_level.value
            if hasattr(risk_record.risk_level, "value")
            else str(risk_record.risk_level)
        )
        return {
            "risk_level": risk_str,
            "risk_class_idx": cls.RISK_MAP.get(risk_str.upper(), -1),
            "risk_factors": risk_record.risk_factors,
            "has_risk_label": True,
            "gat_target_status": cls.GAT_TASK_STATUS,
        }

    @classmethod
    def build_claim_relation_labels(
        cls,
        evidence_nodes: List[EvidenceNode],
    ) -> Dict[str, Any]:
        """Builds claim-level evidence verification relation labels."""
        relations = []
        for ev in evidence_nodes:
            rel_str = (
                ev.relation.value
                if hasattr(ev.relation, "value")
                else str(ev.relation)
            )
            relations.append({
                "evidence_id": ev.evidence_id,
                "relation": rel_str,
                "relation_idx": cls.RELATION_MAP.get(rel_str.upper(), -1),
            })

        return {
            "evidence_relation_count": len(relations),
            "evidence_relations": relations,
            "gat_target_status": cls.GAT_TASK_STATUS,
        }
