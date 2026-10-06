import time
from typing import List, Dict, Any
from pydantic import BaseModel
from app.input.schemas import NormalizedPost
from app.claims.schemas import Claim, AtomicClaim
from app.query_synthesis.schemas import SearchQuery
from app.claims.extractor import RuleBasedClaimExtractor
from app.claims.decomposer import RuleBasedDecomposer
from app.query_synthesis.query_generator import RuleBasedQuerySynthesizer

class ClaimPipelineResult(BaseModel):
    post_id: str
    claims: List[Claim]
    atomic_claims: List[AtomicClaim]
    search_queries: List[SearchQuery]
    processing_time_ms: float
    pipeline_version: str = "1.0.0"
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}

class TrustLensClaimPipeline:
    def __init__(self):
        self.extractor = RuleBasedClaimExtractor()
        self.decomposer = RuleBasedDecomposer()
        self.synthesizer = RuleBasedQuerySynthesizer()
        
    def process(self, post: NormalizedPost) -> ClaimPipelineResult:
        start_time = time.time()
        warnings = []
        
        # Step 11: Claim Extraction
        claims = []
        try:
            claims = self.extractor.extract_from_post(post)
            if not claims:
                warnings.append("No claims extracted from post.")
        except Exception as e:
            warnings.append(f"Claim extraction failed: {str(e)}")
            
        # Step 12: Claim Decomposition
        atomic_claims = []
        try:
            for claim in claims:
                decomposed = self.decomposer.decompose(claim)
                if decomposed.decomposition_status == "failed":
                    warnings.append(f"Decomposition failed for claim {claim.claim_id}")
                atomic_claims.extend(decomposed.atomic_claims)
        except Exception as e:
            warnings.append(f"Claim decomposition failed: {str(e)}")
            
        # Step 13: Query Synthesis
        search_queries = []
        try:
            search_queries = self.synthesizer.synthesize(atomic_claims)
            if not search_queries and atomic_claims:
                warnings.append("No queries generated despite having atomic claims.")
        except Exception as e:
            warnings.append(f"Query synthesis failed: {str(e)}")
            
        processing_time_ms = (time.time() - start_time) * 1000
        
        return ClaimPipelineResult(
            post_id=post.post_id if post else "unknown",
            claims=claims,
            atomic_claims=atomic_claims,
            search_queries=search_queries,
            processing_time_ms=processing_time_ms,
            warnings=warnings,
            metadata={
                "extracted_count": len(claims),
                "atomic_count": len(atomic_claims),
                "queries_count": len(search_queries),
                "extraction_method": "rule_based",
                "decomposition_method": "rule_based",
                "query_synthesis_method": "rule_based"
            }
        )
