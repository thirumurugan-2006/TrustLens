import pytest
from app.claims.schemas import AtomicClaim, SourceSpan
from app.query_synthesis.query_generator import RuleBasedQuerySynthesizer

@pytest.fixture
def synthesizer():
    return RuleBasedQuerySynthesizer()

def make_atomic_claim(text: str, subj: str=None, pred: str=None, obj: str=None, val: str=None, unit: str=None, ctype: str="FACTUAL", verifiable: bool=True, language="en") -> AtomicClaim:
    return AtomicClaim(
        parent_claim_id="p1",
        text=text,
        subject=subj,
        predicate=pred,
        object=obj,
        value=val,
        unit=unit,
        claim_type=ctype,
        language=language,
        verifiable=verifiable,
        source_span=SourceSpan(start=0, end=len(text), source_sentence=text, source_index=0)
    )

def test_factual_temporal_query(synthesizer):
    claim = make_atomic_claim("ABC Investments was founded in 2018.", subj="ABC Investments", pred="founded", val="2018", unit="year")
    queries = synthesizer.synthesize([claim])
    assert len(queries) == 1
    assert queries[0].query_type == "TEMPORAL_VERIFICATION"
    assert "founded 2018" in queries[0].query_text

def test_financial_guarantee_query(synthesizer):
    claim = make_atomic_claim("ABC Wealth guarantees 20% monthly returns.", subj="ABC Wealth", pred="guarantees", val="20", unit="percent", ctype="GUARANTEE")
    queries = synthesizer.synthesize([claim])
    assert len(queries) == 1
    assert queries[0].query_type == "FINANCIAL_VERIFICATION"
    assert queries[0].priority == "HIGH"
    assert "20 percent" in queries[0].query_text

def test_registration_query(synthesizer):
    claim = make_atomic_claim("ABC Wealth is not government approved.", subj="ABC Wealth", pred="government_approved", ctype="FACTUAL")
    claim.negated = True
    queries = synthesizer.synthesize([claim])
    # Should neutral verification without scam/fake
    assert len(queries) == 1
    assert queries[0].query_type == "REGISTRATION_VERIFICATION"
    assert "scam" not in queries[0].query_text.lower()
    assert "government approval registration" in queries[0].query_text.lower()

def test_opinion_no_query(synthesizer):
    claim = make_atomic_claim("I think this investment is amazing.", ctype="OPINION", verifiable=False)
    queries = synthesizer.synthesize([claim])
    assert len(queries) == 0

def test_question_query(synthesizer):
    claim = make_atomic_claim("Is ABC Wealth really government approved?", subj="ABC Wealth", pred="government_approved", ctype="QUESTION", verifiable=False)
    queries = synthesizer.synthesize([claim])
    assert len(queries) == 1
    assert queries[0].query_type == "REGISTRATION_VERIFICATION"
