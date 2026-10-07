"""
TrustLens Evidence Feature Extractor (Phase 6D).
Extracts grounded evidence volume, coverage, source count, and source diversity features.
"""

from typing import Any, Dict, List, Set
from urllib.parse import urlparse

from app.training.schemas import TrainingClaim, TrainingEvidence


class EvidenceFeatureExtractor:
    """Extracts evidence volume and coverage features."""

    @classmethod
    def extract(
        cls,
        claims: List[TrainingClaim],
        evidence_list: List[TrainingEvidence],
    ) -> Dict[str, Any]:
        """Extracts evidence coverage and source diversity metrics."""
        ev_count = len(evidence_list)
        claim_count = len(claims)

        if ev_count == 0:
            return {
                "evidence_count": 0,
                "has_evidence": False,
                "claims_with_evidence": 0,
                "claims_without_evidence": claim_count,
                "evidence_coverage_ratio": 0.0,
                "evidence_source_count": 0,
                "independent_source_count": 0,
                "source_diversity": 0.0,
            }

        # Claims covered
        ev_claim_ids = {e.claim_id for e in evidence_list}
        claims_with_ev = sum(1 for c in claims if c.claim_id in ev_claim_ids)
        claims_without_ev = max(claim_count - claims_with_ev, 0)
        coverage_ratio = round(claims_with_ev / max(claim_count, 1), 4)

        # Sources and domains
        source_types: Set[str] = set()
        domains: Set[str] = set()

        for e in evidence_list:
            if e.source_type:
                source_types.add(e.source_type)
            if e.source_url:
                try:
                    netloc = urlparse(e.source_url).netloc.lower()
                    if netloc:
                        domains.add(netloc)
                except Exception:
                    pass

        source_count = max(len(source_types), 1)
        indep_sources = max(len(domains), 1)
        source_diversity = round(indep_sources / max(ev_count, 1), 4)

        return {
            "evidence_count": ev_count,
            "has_evidence": True,
            "claims_with_evidence": claims_with_ev,
            "claims_without_evidence": claims_without_ev,
            "evidence_coverage_ratio": coverage_ratio,
            "evidence_source_count": source_count,
            "independent_source_count": indep_sources,
            "source_diversity": source_diversity,
        }
