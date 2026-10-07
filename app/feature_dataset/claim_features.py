"""
TrustLens Claim Feature Extractor (Phase 6D).
Derives grounded semantic, atomic decomposition, and factual assertion features from claims.
"""

from typing import Any, Dict, List

from app.training.schemas import ClaimType, TrainingClaim


class ClaimFeatureExtractor:
    """Extracts claim counts, types, verifiability, and semantic indicators."""

    GUARANTEE_TERMS = ["guarantee", "guaranteed", "100%", "zero risk", "urudhi", "pakka", "sure shot"]

    @classmethod
    def extract(cls, claims: List[TrainingClaim]) -> Dict[str, Any]:
        """Extracts claim-level features from associated TrainingClaim list."""
        if not claims:
            return {
                "claim_count": 0,
                "atomic_claim_count": 0,
                "verifiable_claim_count": 0,
                "non_verifiable_claim_count": 0,
                "claim_decomposition_confidence": 0.0,
                "has_financial_claim": False,
                "has_job_claim": False,
                "has_giveaway_claim": False,
                "has_credential_claim": False,
                "has_shopping_claim": False,
                "conditional_claim_count": 0,
                "negated_claim_count": 0,
                "numeric_claim_count": 0,
                "temporal_claim_count": 0,
                "guarantee_claim_present": False,
                "urgency_signal_present": False,
                "upfront_payment_present": False,
                "financial_claim_count": 0,
                "currency_present": False,
                "temporal_deadline_present": False,
            }

        claim_count = len(claims)
        atomic_claim_count = 0
        verifiable_count = 0
        non_verifiable_count = 0
        conf_sum = 0.0

        has_financial = False
        has_job = False
        has_giveaway = False
        has_credential = False
        has_shopping = False
        financial_claims_count = 0

        conditional_count = 0
        negated_count = 0
        numeric_count = 0
        temporal_count = 0
        guarantee_present = False
        currency_present = False
        deadline_present = False

        for c in claims:
            ctype_str = (
                c.claim_type.value
                if hasattr(c.claim_type, "value")
                else str(c.claim_type)
            ).upper()

            if ctype_str == "FINANCIAL":
                has_financial = True
                financial_claims_count += 1
            elif ctype_str == "JOB":
                has_job = True
            elif ctype_str == "GIVEAWAY":
                has_giveaway = True
            elif ctype_str == "CREDENTIAL":
                has_credential = True
            elif ctype_str == "SHOPPING":
                has_shopping = True

            c_text_lower = (c.claim_text or "").lower()
            if any(term in c_text_lower for term in cls.GUARANTEE_TERMS):
                guarantee_present = True

            if any(sym in c.claim_text for sym in ["₹", "$", "Rs", "INR", "USD"]):
                currency_present = True

            if any(word in c_text_lower for word in ["daily", "weekly", "monthly", "hour", "deadline", "today"]):
                deadline_present = True

            # Decomposition / check-worthiness
            conf = float(c.check_worthiness) if c.check_worthiness is not None else 1.0
            conf_sum += conf

            # In our dataset each claim has at least 1 atomic predicate
            atomic_claim_count += 1
            if conf >= 0.5:
                verifiable_count += 1
            else:
                non_verifiable_count += 1

            # Semantic flags
            if any(num in c.claim_text for num in "0123456789%"):
                numeric_count += 1
            if any(kw in c_text_lower for kw in ["if", "deposit", "min", "invest"]):
                conditional_count += 1
            if any(neg in c_text_lower for neg in ["not", "no", "never", "illai", "nahi"]):
                negated_count += 1
            if any(temp in c_text_lower for temp in ["weekly", "monthly", "daily", "days", "hours"]):
                temporal_count += 1

        avg_conf = round(conf_sum / claim_count, 4)

        return {
            "claim_count": claim_count,
            "atomic_claim_count": atomic_claim_count,
            "verifiable_claim_count": verifiable_count,
            "non_verifiable_claim_count": non_verifiable_count,
            "claim_decomposition_confidence": avg_conf,
            "has_financial_claim": has_financial,
            "has_job_claim": has_job,
            "has_giveaway_claim": has_giveaway,
            "has_credential_claim": has_credential,
            "has_shopping_claim": has_shopping,
            "conditional_claim_count": conditional_count,
            "negated_claim_count": negated_count,
            "numeric_claim_count": numeric_count,
            "temporal_claim_count": temporal_count,
            "guarantee_claim_present": guarantee_present,
            "urgency_signal_present": any(any(u in c.claim_text.lower() for u in ["urgent", "today only", "hurry", "limited", "fast", "now", "quick"]) for c in claims),
            "upfront_payment_present": any(any(u in c.claim_text.lower() for u in ["fee", "deposit", "advance", "registration", "processing", "upfront", "pay first"]) for c in claims),
            "financial_claim_count": financial_claims_count,
            "currency_present": currency_present,
            "temporal_deadline_present": deadline_present,
        }
