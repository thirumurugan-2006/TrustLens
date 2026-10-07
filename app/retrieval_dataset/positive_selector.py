from typing import Dict, List, Optional

from app.training.schemas import EvidenceRelationLabel, TrainingEvidence


class PositiveEvidenceSelector:
    """
    Selects valid positive evidence documents for a given claim.
    Enforces:
    1. Positive relation MUST be SUPPORTS or CONTRADICTS.
    2. NEUTRAL and INSUFFICIENT evidence are strictly barred from being positive ground truth.
    3. Claim ID linkage and full provenance preservation.
    """

    @staticmethod
    def is_valid_positive(evidence: TrainingEvidence) -> bool:
        """Checks if an evidence document qualifies as positive ground truth."""
        if not evidence.evidence_text or not evidence.evidence_text.strip():
            return False
        # Must be SUPPORTS or CONTRADICTS
        if evidence.relation_label not in (EvidenceRelationLabel.SUPPORTS, EvidenceRelationLabel.CONTRADICTS):
            return False
        return True

    def select_positive(
        self,
        claim_id: str,
        evidence_list: List[TrainingEvidence],
    ) -> Optional[TrainingEvidence]:
        """
        Finds the primary ground truth positive evidence for a claim_id.
        Prefers evidence directly linked to the claim_id that satisfies positive criteria.
        """
        matching = [e for e in evidence_list if e.claim_id == claim_id]
        for e in matching:
            if self.is_valid_positive(e):
                return e
        return None

    def build_positive_index(
        self,
        evidence_list: List[TrainingEvidence],
    ) -> Dict[str, TrainingEvidence]:
        """Builds a map from claim_id -> primary valid positive TrainingEvidence."""
        index: Dict[str, TrainingEvidence] = {}
        for e in evidence_list:
            if self.is_valid_positive(e):
                if e.claim_id not in index:
                    index[e.claim_id] = e
        return index
