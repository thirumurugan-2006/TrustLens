"""
TrustLens Source Feature Extractor (Phase 6D).
Extracts observable evidence citation sources and regulatory alert presence.
"""

from typing import Any, Dict, List, Set

from app.training.schemas import ProvenanceSourceType, TrainingEvidence


class SourceFeatureExtractor:
    """Extracts evidence source volume and authority features."""

    OFFICIAL_SOURCE_TYPES = {"REGULATORY_ALERT", "PUBLIC_DATASET", "FACT_CHECKING"}

    @classmethod
    def extract(cls, evidence_list: List[TrainingEvidence]) -> Dict[str, Any]:
        """Extracts source features from evidence items."""
        if not evidence_list:
            return {
                "source_count": 0,
                "unique_source_count": 0,
                "official_source_present": False,
            }

        source_names: Set[str] = set()
        source_ids: Set[str] = set()
        official_present = False

        for e in evidence_list:
            stype = (e.source_type or "").upper()
            if stype in cls.OFFICIAL_SOURCE_TYPES or "regulatory" in stype.lower() or "official" in stype.lower():
                official_present = True

            if e.provenance:
                if e.provenance.source_name:
                    source_names.add(e.provenance.source_name)
                if e.provenance.source_id:
                    source_ids.add(e.provenance.source_id)
            elif e.source_title:
                source_names.add(e.source_title)

        source_count = max(len(source_names), 1)
        unique_source_count = max(len(source_ids), len(source_names), 1)

        return {
            "source_count": source_count,
            "unique_source_count": unique_source_count,
            "official_source_present": official_present,
        }
