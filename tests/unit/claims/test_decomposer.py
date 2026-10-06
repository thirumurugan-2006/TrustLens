import pytest
from app.claims.schemas import Claim, SourceSpan
from app.claims.decomposer import RuleBasedDecomposer

@pytest.fixture
def decomposer():
    return RuleBasedDecomposer()

def make_claim(text: str, ctype: str = "FACTUAL", meta: dict = None, language: str = "en") -> Claim:
    if meta is None:
        meta = {}
    return Claim(
        text=text,
        normalized_text=text,
        claim_type=ctype,
        language=language,
        source_span=SourceSpan(start=0, end=len(text), source_sentence=text, source_index=0),
        extraction_method="rule_based",
        verifiable=True,
        metadata=meta
    )

def test_factual_decomposition(decomposer):
    claim = make_claim("ABC Investments was founded in 2018.")
    res = decomposer.decompose(claim)
    assert len(res.atomic_claims) == 1
    ac = res.atomic_claims[0]
    assert ac.subject == "ABC Investments"
    assert ac.predicate == "founded"
    assert ac.value == "2018"
    assert ac.unit == "year"

def test_two_factual_claims(decomposer):
    claim = make_claim("ABC Investments was founded in 2018 and has 50,000 customers.")
    res = decomposer.decompose(claim)
    assert len(res.atomic_claims) == 2
    assert res.atomic_claims[0].subject == "ABC Investments"
    # test simple coreference via state across parts (although within same sentence)
    assert res.atomic_claims[1].subject == "ABC Investments"
    assert res.atomic_claims[1].predicate == "has"
    assert res.atomic_claims[1].value == "50000"

def test_financial_guarantee_claim(decomposer):
    claim = make_claim("ABC Wealth guarantees 20% monthly returns.", meta={"percentages": ["20%"]})
    res = decomposer.decompose(claim)
    assert len(res.atomic_claims) == 1
    ac = res.atomic_claims[0]
    assert ac.subject == "ABC Wealth"
    assert ac.predicate == "guarantees"
    assert ac.value == "20"
    assert ac.unit == "percent"
    assert ac.temporal_context["type"] == "frequency"

def test_conditional_financial(decomposer):
    claim = make_claim("If you invest ₹5,000 today, you will receive ₹50,000 within 30 days.", meta={"numbers": ["5000", "50000"]})
    res = decomposer.decompose(claim)
    assert len(res.atomic_claims) == 2
    assert res.atomic_claims[0].predicate == "invest"
    assert res.atomic_claims[0].value == "5000"
    assert res.atomic_claims[1].predicate == "receive"
    assert res.atomic_claims[1].value == "50000"

def test_negation(decomposer):
    claim = make_claim("ABC Wealth is not government approved.")
    res = decomposer.decompose(claim)
    assert res.atomic_claims[0].subject == "ABC Wealth"
    assert res.atomic_claims[0].predicate == "government_approved"
    assert res.atomic_claims[0].negated is True
    assert res.atomic_claims[0].polarity == "NEGATIVE"

def test_question(decomposer):
    claim = make_claim("Is ABC Wealth really government approved?", ctype="QUESTION")
    res = decomposer.decompose(claim)
    assert res.atomic_claims[0].subject == "ABC Wealth"
    assert res.atomic_claims[0].predicate == "government_approved"
    assert res.atomic_claims[0].claim_type == "QUESTION"

def test_coreference_across_claims(decomposer):
    claim1 = make_claim("The company has 50,000 customers.")
    claim2 = make_claim("Its headquarters are in Chennai.")
    # seed
    decomposer._last_subject = "ABC Investments"
    res1 = decomposer.decompose(claim1)
    res2 = decomposer.decompose(claim2)
    assert res1.atomic_claims[0].subject == "ABC Investments"
    assert res2.atomic_claims[0].subject == "ABC Investments"
