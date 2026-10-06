import pytest
from app.pipeline.claim_pipeline import TrustLensClaimPipeline
from app.input.schemas import NormalizedPost

@pytest.fixture
def pipeline():
    return TrustLensClaimPipeline()

def make_post(text: str, lang: str = "en") -> NormalizedPost:
    return NormalizedPost(
        post_id="post_123",
        platform="test",
        source_url="http://test",
        author="test_author",
        title="test_title",
        text=text,
        images=[],
        comments=[],
        timestamp="2023-01-01T00:00:00Z",
        metadata={"language_analysis": {"primary_language": lang, "is_code_mixed": False}}
    )

def test_full_pipeline_complex(pipeline):
    text = "ABC Investments was founded in 2018. The company has 50,000 customers. Its headquarters are in Chennai."
    post = make_post(text)
    result = pipeline.process(post)
    
    assert len(result.claims) == 3
    assert len(result.atomic_claims) == 3
    
    # coreference test
    assert result.atomic_claims[0].subject == "ABC Investments"
    assert result.atomic_claims[1].subject == "ABC Investments"
    assert result.atomic_claims[2].subject == "ABC Investments"
    
    assert len(result.search_queries) == 3

def test_full_pipeline_conditional(pipeline):
    text = "If you invest ₹5,000 today, you will receive ₹50,000 within 30 days."
    post = make_post(text)
    result = pipeline.process(post)
    assert len(result.claims) == 1
    assert len(result.atomic_claims) == 2
    assert result.atomic_claims[0].metadata.get("relationship") == "condition -> outcome"

def test_full_pipeline_no_claims(pipeline):
    post = make_post("   ")
    result = pipeline.process(post)
    assert len(result.claims) == 0
    assert len(result.atomic_claims) == 0
    assert len(result.search_queries) == 0
