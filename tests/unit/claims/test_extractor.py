import pytest
from app.claims.extractor import RuleBasedClaimExtractor
from app.input.schemas import NormalizedPost, Platform, PostType, AuthorInfo, PostContent, PostMedia

@pytest.fixture
def extractor():
    return RuleBasedClaimExtractor()

def make_post(text: str) -> NormalizedPost:
    return NormalizedPost(
        post_id="test_post",
        platform=Platform.generic,
        post_type=PostType.text,
        content=PostContent(text=text if text is not None else ""),
        media=PostMedia(),
        source_url="http://test",
        author=AuthorInfo(),
        comments=[],
        timestamp="2023-01-01T00:00:00Z",
        metadata={"language_analysis": {"primary_language": "en"}}
    )

def test_english_factual_claim(extractor):
    post = make_post("The bank requires identity verification.")
    claims = extractor.extract_from_post(post)
    assert len(claims) == 1
    assert claims[0].claim_type == "FACTUAL"
    assert claims[0].verifiable is True

def test_financial_guarantee_claim(extractor):
    post = make_post("This investment guarantees 20% monthly returns.")
    claims = extractor.extract_from_post(post)
    assert len(claims) == 1
    assert claims[0].claim_type == "GUARANTEE"
    assert claims[0].verifiable is True
    assert "20%" in claims[0].metadata["percentages"]
    assert "20" in claims[0].metadata["numbers"]

def test_request_instruction(extractor):
    post = make_post("DM me to join. Share this post.")
    claims = extractor.extract_from_post(post)
    types = [c.claim_type for c in claims]
    assert "REQUEST" in types or "INSTRUCTION" in types
    assert not any(c.verifiable for c in claims)

def test_contextual(extractor):
    post = make_post("I received this message yesterday.")
    claims = extractor.extract_from_post(post)
    assert claims[0].claim_type == "CONTEXTUAL"

def test_question(extractor):
    post = make_post("Is this website legitimate?")
    claims = extractor.extract_from_post(post)
    assert claims[0].claim_type == "QUESTION"

def test_tamil_claim(extractor):
    post = make_post("இந்த investment மாதம் 20% return guarantee செய்கிறது.")
    claims = extractor.extract_from_post(post)
    assert len(claims) == 1
    assert claims[0].claim_type == "GUARANTEE"
    assert "20%" in claims[0].metadata["percentages"]

def test_tanglish_claim(extractor):
    post = make_post("Indha investment monthly 20% return guarantee pannum.")
    claims = extractor.extract_from_post(post)
    assert len(claims) == 1
    assert claims[0].claim_type == "GUARANTEE"
    assert "20%" in claims[0].metadata["percentages"]

def test_duplicate_claims(extractor):
    post = make_post("Earn 50k! Earn 50k! earn 50k!")
    claims = extractor.extract_from_post(post)
    assert len(claims) == 1

def test_url_and_decimals(extractor):
    post = make_post("Earn $500.50/month. Apply at example.com.")
    claims = extractor.extract_from_post(post)
    assert len(claims) > 0
    assert "$500.50" in claims[0].metadata["currency"]

def test_empty_text(extractor):
    post = make_post("")
    claims = extractor.extract_from_post(post)
    assert len(claims) == 0

    post2 = make_post("   ")
    claims2 = extractor.extract_from_post(post2)
    assert len(claims2) == 0

def test_unreliable_ocr(extractor):
    post = make_post(None)
    claims = extractor.extract_from_post(post)
    assert len(claims) == 0

def test_source_span(extractor):
    text = "First claim. Second claim."
    post = make_post(text)
    claims = extractor.extract_from_post(post)
    assert len(claims) == 2
    assert claims[0].source_span.start == 0
    assert claims[1].source_span.start > 0
    
def test_no_scam_classification(extractor):
    post = make_post("Invest 5000 get 50000 fraud scam")
    claims = extractor.extract_from_post(post)
    assert "scam" not in claims[0].claim_type.lower()

def test_attribution_claim(extractor):
    post = make_post("According to the company, investors can earn 20% monthly returns.")
    claims = extractor.extract_from_post(post)
    assert len(claims) == 1
    assert "attribution" in claims[0].metadata
    assert claims[0].metadata["attribution"]["source"].lower() == "the company"
