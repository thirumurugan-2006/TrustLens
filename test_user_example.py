from app.pipeline.claim_pipeline import TrustLensClaimPipeline
from app.input.schemas import NormalizedPost, Platform, PostType, AuthorInfo, PostContent, PostMedia
import json

def make_post(text: str) -> NormalizedPost:
    return NormalizedPost(
        post_id="post_123",
        platform=Platform.generic,
        post_type=PostType.text,
        content=PostContent(text=text),
        media=PostMedia(),
        source_url="http://test",
        author=AuthorInfo(),
        comments=[],
        timestamp="2023-01-01T00:00:00Z",
        metadata={"language_analysis": {"primary_language": "en", "is_code_mixed": False}}
    )

pipeline = TrustLensClaimPipeline()
text = "ABC Wealth guarantees 20% monthly returns. Invest ₹10,000 today and receive ₹20,000 within 30 days."
post = make_post(text)

result = pipeline.process(post)
print(json.dumps(result, indent=2, default=str))
