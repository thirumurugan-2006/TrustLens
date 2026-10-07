import re
import uuid
from typing import Any, Dict, List, Optional

from app.claims.schemas import AtomicClaim
from app.query_synthesis.query_generator import RuleBasedQuerySynthesizer
from app.training.schemas import (
    ClaimType,
    QueryType,
    TrainingClaim,
)


class RetrievalQueryBuilder:
    """
    Synthesizes clean, claim-specific, retrieval-oriented queries from TrainingClaims.
    Enforces:
    1. Traceability to post_id, claim_id, atomic_claim_id.
    2. Prevention of verbatim social media dumps or injected scam tropes.
    3. Multi-variant generation (primary verification, entity, claim keywords).
    4. Language and script preservation across English, Tamil, Hindi, Tanglish, and Hinglish.
    """

    def __init__(self):
        self.synthesizer = RuleBasedQuerySynthesizer()

    @staticmethod
    def clean_query_text(text: str) -> str:
        """Strips noise, excessive punctuation, and formatting while preserving keywords."""
        cleaned = re.sub(r"https?://\S+", "", text)
        cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
        cleaned = re.sub(r"[!?,:;\"'\\(\\)\\[\\]\\{\\}]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def extract_keywords_from_claim(self, claim_text: str, language: str) -> str:
        """Extracts retrieval-oriented keywords from claim text without copying entire posts."""
        cleaned = self.clean_query_text(claim_text)
        words = cleaned.split()

        # If claim is already concise (<= 8 words), return as-is
        if len(words) <= 8:
            return cleaned

        # Filter stop phrases and contact handles
        filtered_words = []
        for w in words:
            w_lower = w.lower()
            if any(p in w_lower for p in ["+91", "@okhdfc", "@okaxis", "@paytm", "portal-", ".org", ".com", ".net", ".top"]):
                continue
            filtered_words.append(w)

        # Retain first 8-10 informative words
        summary = " ".join(filtered_words[:10])
        return summary if summary else cleaned

    def build_queries_for_claim(
        self,
        claim: TrainingClaim,
        atomic_claims: Optional[List[AtomicClaim]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Builds 1 to 3 distinct, high-quality query variants for a single TrainingClaim.
        Returns list of query candidate dicts.
        """
        queries: List[Dict[str, Any]] = []
        c_text = claim.claim_text
        lang = claim.language or "en"
        c_type = claim.claim_type.value if hasattr(claim.claim_type, "value") else str(claim.claim_type)

        # 1. Primary Verification Query
        q_text_primary = self.extract_keywords_from_claim(c_text, lang)
        q_type_primary = QueryType.VERIFICATION
        if c_type.upper() in ("FINANCIAL", "INVESTMENT"):
            q_type_primary = QueryType.FINANCIAL
        elif c_type.upper() == "TEMPORAL":
            q_type_primary = QueryType.TEMPORAL

        queries.append({
            "query_id": f"q_{claim.claim_id}_01",
            "post_id": claim.post_id,
            "claim_id": claim.claim_id,
            "atomic_claim_id": f"atomic_{claim.claim_id}_01",
            "query_text": q_text_primary,
            "query_type": q_type_primary,
            "language": lang,
            "generation_method": "claim_summary_verification",
            "variant_name": "primary_verification",
        })

        # 2. Entity or Topical Query Variant
        # Identify key subject / entity terms
        words = q_text_primary.split()
        if len(words) >= 4:
            entity_query_text = " ".join(words[:5]) + " official registration"
            queries.append({
                "query_id": f"q_{claim.claim_id}_02",
                "post_id": claim.post_id,
                "claim_id": claim.claim_id,
                "atomic_claim_id": f"atomic_{claim.claim_id}_01",
                "query_text": entity_query_text,
                "query_type": QueryType.ENTITY,
                "language": lang,
                "generation_method": "entity_registration_focus",
                "variant_name": "entity_variant",
            })

        # 3. RuleBasedQuerySynthesizer Grouping if atomic claim is provided
        if atomic_claims:
            for ac in atomic_claims:
                syn_queries = self.synthesizer._build_queries(ac)
                for sq in syn_queries:
                    queries.append({
                        "query_id": f"q_{claim.claim_id}_{len(queries)+1:02d}",
                        "post_id": claim.post_id,
                        "claim_id": claim.claim_id,
                        "atomic_claim_id": sq.atomic_claim_id,
                        "query_text": sq.query_text,
                        "query_type": QueryType.CLAIM,
                        "language": sq.language,
                        "generation_method": "rule_based_atomic_synthesis",
                        "variant_name": "atomic_claim_variant",
                    })

        return queries
