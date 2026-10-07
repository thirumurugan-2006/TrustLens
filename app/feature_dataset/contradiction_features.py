"""
TrustLens Contradiction & Stance Feature Extractor (Phase 6D).
Computes support, contradiction, neutral, and insufficient evidence ratios and stance entropy.
"""

import math
from typing import Any, Dict, List

from app.training.schemas import EvidenceRelationLabel, TrainingEvidence


class ContradictionFeatureExtractor:
    """Extracts claim verification stance and contradiction distribution features."""

    @classmethod
    def extract(cls, evidence_list: List[TrainingEvidence]) -> Dict[str, Any]:
        """Computes contradiction and stance distributions from evidence records."""
        ev_count = len(evidence_list)
        if ev_count == 0:
            return {
                "support_count": 0,
                "contradict_count": 0,
                "neutral_count": 0,
                "insufficient_count": 0,
                "support_ratio": 0.0,
                "contradiction_ratio": 0.0,
                "neutral_ratio": 0.0,
                "insufficient_ratio": 0.0,
                "support_to_contradiction_ratio": 0.0,
                "contradiction_presence": False,
                "support_presence": False,
                "evidence_relation_entropy": 0.0,
            }

        support_count = 0
        contradict_count = 0
        neutral_count = 0
        insufficient_count = 0

        for e in evidence_list:
            rel = e.relation_label
            if rel == EvidenceRelationLabel.CONTRADICTS:
                contradict_count += 1
            elif rel == EvidenceRelationLabel.SUPPORTS:
                support_count += 1
            elif rel == EvidenceRelationLabel.NEUTRAL:
                neutral_count += 1
            elif rel == EvidenceRelationLabel.INSUFFICIENT:
                insufficient_count += 1

        support_ratio = round(support_count / ev_count, 4)
        contradict_ratio = round(contradict_count / ev_count, 4)
        neutral_ratio = round(neutral_count / ev_count, 4)
        insufficient_ratio = round(insufficient_count / ev_count, 4)

        support_to_contradict = round(support_count / (contradict_count + 1), 4)

        # Shannon Entropy over 4 relation classes
        probs = [support_ratio, contradict_ratio, neutral_ratio, insufficient_ratio]
        entropy = 0.0
        for p in probs:
            if p > 0.0:
                entropy -= p * math.log2(p)
        entropy = round(entropy, 4)

        return {
            "support_count": support_count,
            "contradict_count": contradict_count,
            "neutral_count": neutral_count,
            "insufficient_count": insufficient_count,
            "support_ratio": support_ratio,
            "contradiction_ratio": contradict_ratio,
            "neutral_ratio": neutral_ratio,
            "insufficient_ratio": insufficient_ratio,
            "support_to_contradiction_ratio": support_to_contradict,
            "contradiction_presence": bool(contradict_count > 0),
            "support_presence": bool(support_count > 0),
            "evidence_relation_entropy": entropy,
        }
