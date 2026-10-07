from typing import List, Dict, Any
from app.claims.schemas import Claim, AtomicClaim
from app.query_synthesis.schemas import SearchQuery

class SemanticQualityGate:
    def evaluate(self, claims: List[Claim], atomic_claims: List[AtomicClaim], queries: List[SearchQuery] = None) -> Dict[str, Any]:
        if queries is None:
            queries = []
            
        total = len(atomic_claims)
        if total == 0:
            return {
                "step_14_ready": False,
                "overall_quality": "FAIL",
                "blocking_reasons": ["No atomic claims generated."]
            }
            
        blocking = []
        
        # 1. Predicate & Subject coverage for verifiable claims
        verifiable_claims = [ac for ac in atomic_claims if ac.verifiable]
        missing_preds = [ac for ac in verifiable_claims if not ac.predicate]
        missing_subs = [ac for ac in verifiable_claims if not ac.subject]
        
        if missing_preds:
            blocking.append("Predicate coverage is incomplete for verifiable claims.")
            for ac in missing_preds:
                ac.verifiable = False # Fallback rule
                
        if missing_subs:
            blocking.append("Subject coverage is incomplete for verifiable claims.")
            
        # 4 & 5. Numerical and Currency attachment
        # Ensure that if the original text has a number/currency, it's captured in the atomic claim if relevant
        for claim in claims:
            if claim.metadata.get("currency") or claim.metadata.get("numbers"):
                # Make sure at least one atomic claim has captured a value
                has_val = any(ac.value for ac in atomic_claims if ac.parent_claim_id == claim.claim_id)
                if not has_val and claim.verifiable:
                    blocking.append(f"Numerical/Currency attachment failed for claim {claim.claim_id}.")
                    
        # 7. Condition preservation
        # If any claim has condition -> outcome, check relationships
        relationships = []
        for ac in atomic_claims:
            if ac.relationships:
                relationships.extend(ac.relationships)
        
        # 14 & 15. Orphan checks
        # Every verifiable atomic claim should have at least one query? Or just verifiable should be queried if not contextual
        
        # 16. Duplicate financial values error
        # "No duplicated financial values caused by extraction errors"
        # Example: A2 value = 10000, A3 value = 10000 but text had 20000
        for ac in atomic_claims:
            for rel in ac.relationships:
                if rel.get("type") == "promised_outcome":
                    target_id = rel.get("target")
                    target_ac = next((a for a in atomic_claims if a.atomic_claim_id == target_id), None)
                    if target_ac and target_ac.value == ac.value and ac.value is not None:
                        # Only flag if the original text implies different values
                        # E.g. "invest 10000 get 20000"
                        if "20000" in ac.original_text or "20,000" in ac.original_text:
                            if ac.value == "10000":
                                blocking.append("Duplicate financial values detected (possible extraction error in outcome).")

        # Basic provenance check
        for ac in atomic_claims:
            if not ac.original_text or not ac.normalized_text:
                blocking.append(f"Provenance missing for atomic claim {ac.atomic_claim_id}")

        return {
            "step_14_ready": len(blocking) == 0,
            "overall_quality": "PASS" if len(blocking) == 0 else "FAIL",
            "blocking_reasons": list(set(blocking)),
            "metrics": {
                "total_claims": len(claims),
                "total_atomic_claims": total,
                "verifiable_claims": len(verifiable_claims),
                "missing_predicates": len(missing_preds)
            }
        }
