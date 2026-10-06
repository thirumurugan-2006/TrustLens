from typing import List
from app.claims.schemas import AtomicClaim
from app.query_synthesis.schemas import SearchQuery

class RuleBasedQuerySynthesizer:
    def __init__(self):
        self.query_type_map = {
            "FACTUAL": ("DIRECT_VERIFICATION", ["general_web", "reputable_news"]),
            "GUARANTEE": ("FINANCIAL_VERIFICATION", ["official_company", "financial_regulator"]),
            "FINANCIAL": ("FINANCIAL_VERIFICATION", ["official_company", "authoritative_financial"]),
            "NUMERICAL": ("NUMERICAL_VERIFICATION", ["official_company"]),
            "REGISTRATION": ("REGISTRATION_VERIFICATION", ["official_government", "regulatory_databases"]),
            "QUESTION": ("REGISTRATION_VERIFICATION", ["official_government"])
        }

    def _determine_priority(self, claim_type: str) -> str:
        if claim_type in ["GUARANTEE", "FINANCIAL", "REGISTRATION", "NUMERICAL"]:
            return "HIGH"
        elif claim_type in ["OPINION", "REQUEST", "INSTRUCTION", "CONTEXTUAL"]:
            return "LOW"
        return "HIGH"

    def _build_queries(self, atomic_claim: AtomicClaim) -> List[SearchQuery]:
        queries = []
        text = atomic_claim.text
        claim_type = atomic_claim.claim_type
        
        if not atomic_claim.verifiable and claim_type != "QUESTION":
            return []

        q_type, default_sources = self.query_type_map.get(claim_type, ("GENERAL_CONTEXT", ["general_web"]))
        priority = self._determine_priority(claim_type)

        subj = atomic_claim.subject or ""
        pred = atomic_claim.predicate or ""
        val = atomic_claim.value or ""
        unit = atomic_claim.unit or ""
        obj = atomic_claim.object or ""

        base_query = text

        if "government" in text.lower() or "approv" in text.lower() or "register" in text.lower():
            q_type = "REGISTRATION_VERIFICATION"
            priority = "HIGH"
            default_sources = ["official_government", "regulatory_databases"]
            if subj:
                base_query = f"{subj} government approval registration"
            else:
                base_query = "government approval registration"

        elif claim_type == "FACTUAL" and pred == "founded" and subj:
            q_type = "TEMPORAL_VERIFICATION"
            base_query = f"{subj} founded {val}"
        elif claim_type == "FACTUAL" and pred == "headquartered_in" and subj:
            q_type = "ENTITY_VERIFICATION"
            base_query = f"{subj} headquarters {obj}"
        elif claim_type == "NUMERICAL" and subj and val:
            q_type = "NUMERICAL_VERIFICATION"
            base_query = f"{subj} {val} {unit}".strip()
        elif claim_type == "GUARANTEE" and subj:
            q_type = "FINANCIAL_VERIFICATION"
            base_query = f"{subj} {val} percent {atomic_claim.temporal_context.get('value', '') if isinstance(atomic_claim.temporal_context, dict) else ''} returns"
        elif claim_type == "FINANCIAL" and "invest" in text.lower():
             base_query = text
             
        queries.append(SearchQuery(
            atomic_claim_id=atomic_claim.atomic_claim_id,
            query_text=base_query.strip(),
            query_type=q_type,
            language=atomic_claim.language,
            priority=priority,
            source_preferences=default_sources
        ))
        
        return queries

    def synthesize(self, atomic_claims: List[AtomicClaim]) -> List[SearchQuery]:
        all_queries = []
        seen_queries = set()
        
        # Group conditionals
        conditions = [ac for ac in atomic_claims if ac.metadata.get("relationship") == "condition -> outcome" and ac.predicate == "invest"]
        outcomes = [ac for ac in atomic_claims if ac.metadata.get("relationship") == "condition -> outcome" and ac.predicate == "receive"]
        
        paired = set()
        for cond in conditions:
            for out in outcomes:
                if cond.parent_claim_id == out.parent_claim_id and cond.atomic_claim_id not in paired:
                    paired.add(cond.atomic_claim_id)
                    paired.add(out.atomic_claim_id)
                    q_text = f"investment {cond.value} promised return {out.value} {out.temporal_context.get('value', '') if out.temporal_context else ''}".strip()
                    q = SearchQuery(
                        atomic_claim_id=cond.atomic_claim_id,
                        query_text=q_text,
                        query_type="FINANCIAL_VERIFICATION",
                        language=cond.language,
                        priority="HIGH",
                        source_preferences=["authoritative_financial", "official_company"]
                    )
                    all_queries.append(q)
                    seen_queries.add(f"{q.query_text.lower()}|{q.query_type}")

        for ac in atomic_claims:
            if ac.atomic_claim_id in paired:
                continue
            queries = self._build_queries(ac)
            for q in queries:
                key = f"{q.query_text.lower()}|{q.query_type}"
                if key not in seen_queries:
                    seen_queries.add(key)
                    all_queries.append(q)
                    
        return all_queries
