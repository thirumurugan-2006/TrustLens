from typing import List, Dict, Any
from app.claims.schemas import Claim, AtomicClaim

class SemanticQualityGate:
    def evaluate(self, claims: List[Claim], atomic_claims: List[AtomicClaim]) -> Dict[str, Any]:
        total = len(atomic_claims)
        if total == 0:
            return {
                "step_14_ready": False,
                "overall_quality": "FAIL",
                "blocking_reasons": ["No atomic claims generated."]
            }
            
        sub_count = len([a for a in atomic_claims if a.subject])
        pred_count = len([a for a in atomic_claims if a.predicate])
        obj_val_count = len([a for a in atomic_claims if a.object or a.value])
        
        has_temporal = any(a.temporal_context for a in atomic_claims)
        has_rel = any(a.metadata.get("relationship") for a in atomic_claims)
        
        blocking = []
        if sub_count < total:
            blocking.append("Subject coverage is incomplete.")
        if pred_count < total:
            blocking.append("Predicate coverage is incomplete.")
            
        return {
            "step_14_ready": len(blocking) == 0,
            "overall_quality": "PASS" if len(blocking) == 0 else "FAIL",
            "blocking_reasons": blocking,
            "metrics": {
                "subject_coverage": f"{sub_count}/{total}",
                "predicate_coverage": f"{pred_count}/{total}",
                "object_value_coverage": f"{obj_val_count}/{total}"
            }
        }
