from app.claims.schemas import Claim, SourceSpan

def test_claim_schema():
    claim = Claim(
        text="test",
        normalized_text="test",
        claim_type="FACTUAL",
        language="en"
    )
    assert claim.claim_id is not None
    assert claim.extraction_method == "rule_based"
    assert claim.verifiable is False
