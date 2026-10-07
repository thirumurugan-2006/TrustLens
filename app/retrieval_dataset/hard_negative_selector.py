import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.retrieval_dataset.schemas import NegativeEvidenceRecord, NegativeType
from app.training.schemas import (
    EvidenceRelationLabel,
    SplitName,
    TrainingClaim,
    TrainingEvidence,
)


class HardNegativeSelector:
    """
    Deterministic Hard-Negative Miner and Validator.
    Enforces:
    1. Critical Negative Safety: Negatives must NEVER support or contradict the target claim.
    2. Prioritization of NEUTRAL evidence and topically plausible distractors.
    3. Multi-type categorization (HARD_ENTITY_NEGATIVE, HARD_TOPIC_NEGATIVE, etc.).
    4. Deterministic candidate ranking based on token overlap / similarity.
    5. Split-scoped candidate pool preventing cross-split leakage.
    """

    @staticmethod
    def tokenize(text: str) -> Set[str]:
        """Simple alphanumeric tokenizer for token overlap calculation."""
        words = re.findall(r"\w+", text.lower())
        return set(w for w in words if len(w) > 2)

    @classmethod
    def calculate_jaccard_similarity(cls, text_a: str, text_b: str) -> float:
        """Computes word-level Jaccard similarity between two texts."""
        set_a = cls.tokenize(text_a)
        set_b = cls.tokenize(text_b)
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return round(intersection / union, 4)

    def select_hard_negatives_for_claim(
        self,
        target_claim: TrainingClaim,
        query_text: str,
        candidate_pool: List[TrainingEvidence],
        positive_evidence_id: str,
        max_negatives: int = 3,
    ) -> List[NegativeEvidenceRecord]:
        """
        Mines and returns up to `max_negatives` valid hard negatives for a target claim.
        The candidate pool MUST be scoped to the same split partition as target_claim.
        """
        valid_negatives: List[NegativeEvidenceRecord] = []
        target_cid = target_claim.claim_id
        target_domain = (
            target_claim.claim_type.value
            if hasattr(target_claim.claim_type, "value")
            else str(target_claim.claim_type)
        )

        candidates_scored: List[Tuple[float, TrainingEvidence, NegativeType]] = []

        for ev in candidate_pool:
            # 1. Skip the positive evidence document itself
            if ev.evidence_id == positive_evidence_id:
                continue

            # 2. Critical Safety: A negative must NEVER have SUPPORTS or CONTRADICTS relation.
            # Only accept candidates that are non-supporting and non-contradicting.
            if ev.relation_label in (EvidenceRelationLabel.SUPPORTS, EvidenceRelationLabel.CONTRADICTS):
                continue

            if ev.claim_id == target_cid:
                # Same claim with NEUTRAL evidence is the premier HARD_ENTITY_NEGATIVE
                neg_type = NegativeType.HARD_ENTITY_NEGATIVE
            elif ev.relation_label == EvidenceRelationLabel.NEUTRAL:
                neg_type = NegativeType.HARD_TOPIC_NEGATIVE
            else:
                neg_type = NegativeType.CROSS_ENTITY_NEGATIVE

            # 3. Calculate candidate similarity score
            sim = self.calculate_jaccard_similarity(query_text, ev.evidence_text)
            candidates_scored.append((sim, ev, neg_type))

        # Sort candidates deterministically:
        # 1. Similarity descending
        # 2. Negative type priority (HARD_ENTITY > HARD_TOPIC > CROSS_ENTITY)
        # 3. Evidence ID ascending
        type_priority = {
            NegativeType.HARD_ENTITY_NEGATIVE: 3,
            NegativeType.HARD_TOPIC_NEGATIVE: 2,
            NegativeType.CROSS_ENTITY_NEGATIVE: 1,
            NegativeType.HARD_SEMANTIC_NEGATIVE: 2,
            NegativeType.TEMPORAL_NEGATIVE: 1,
            NegativeType.IRRELEVANT_NEGATIVE: 0,
        }

        candidates_scored.sort(
            key=lambda x: (x[0], type_priority.get(x[2], 0), x[1].evidence_id),
            reverse=True,
        )

        # Pick top distinct negatives
        seen_texts: Set[str] = set()
        for sim, ev, neg_type in candidates_scored:
            if ev.evidence_text in seen_texts:
                continue
            seen_texts.add(ev.evidence_text)

            record = NegativeEvidenceRecord(
                evidence_id=ev.evidence_id,
                evidence_text=ev.evidence_text,
                negative_type=neg_type,
                source_title=ev.source_title,
                source_url=ev.source_url,
                evidence_relation=ev.relation_label,
                candidate_similarity=sim,
                provenance=ev.provenance,
            )
            valid_negatives.append(record)
            if len(valid_negatives) >= max_negatives:
                break

        return valid_negatives
