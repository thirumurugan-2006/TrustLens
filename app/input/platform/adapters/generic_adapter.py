from app.input.platform.base import BasePlatformAdapter
from app.input.schemas import UniversalSocialPost, Platform, PostType, AcquisitionMethod, AuthorInfo, PostContent
import uuid

class GenericAdapter(BasePlatformAdapter):
    def retrieve(self, identifier: str) -> dict:
        return {"text": identifier}

    def validate(self, raw_data: dict) -> bool:
        return True

    def normalize(self, raw_data: dict) -> UniversalSocialPost:
        return UniversalSocialPost(
            post_id=str(uuid.uuid4()),
            platform=Platform.generic,
            post_type=PostType.text,
            content=PostContent(text=raw_data.get("text", "")),
            acquisition=raw_data.get("acquisition", AcquisitionMethod.copied_text)
        )
