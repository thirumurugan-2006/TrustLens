from app.input.platform.base import BasePlatformAdapter
from app.input.schemas import UniversalSocialPost, Platform, PostType, AcquisitionMethod, AuthorInfo, PostContent
import uuid

class RedditAdapter(BasePlatformAdapter):
    def retrieve(self, identifier: str) -> dict:
        return {"status": "UNAVAILABLE", "reason": "Reddit API not configured"}

    def validate(self, raw_data: dict) -> bool:
        return True

    def normalize(self, raw_data: dict) -> UniversalSocialPost:
        return UniversalSocialPost(
            post_id=str(uuid.uuid4()),
            platform=Platform.reddit,
            post_type=PostType.unknown,
            content=PostContent(text=raw_data.get("text", "")),
            acquisition=AcquisitionMethod.reddit_api
        )
