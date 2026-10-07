from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum

class Platform(str, Enum):
    reddit = "reddit"
    youtube = "youtube"
    x = "x"
    instagram = "instagram"
    facebook = "facebook"
    linkedin = "linkedin"
    tiktok = "tiktok"
    telegram = "telegram"
    whatsapp = "whatsapp"
    generic = "generic"
    text = "text"
    screenshot = "screenshot"
    unknown = "unknown"

class PostType(str, Enum):
    text = "text"
    image = "image"
    video = "video"
    audio = "audio"
    image_text = "image_text"
    video_text = "video_text"
    audio_text = "audio_text"
    carousel = "carousel"
    link = "link"
    poll = "poll"
    thread = "thread"
    comment = "comment"
    reply = "reply"
    unknown = "unknown"

class AcquisitionMethod(str, Enum):
    reddit_api = "reddit_api"
    official_api = "official_api"
    uploaded = "uploaded"
    screenshot = "screenshot"
    copied_text = "copied_text"
    local_dataset = "local_dataset"
    local_cache = "local_cache"
    manual = "manual"
    unknown = "unknown"

class AuthorInfo(BaseModel):
    id: Optional[str] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    verified: bool = False

class PostContent(BaseModel):
    title: Optional[str] = None
    text: str
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)

class PostMedia(BaseModel):
    images: List[str] = Field(default_factory=list)
    videos: List[str] = Field(default_factory=list)
    audio: List[str] = Field(default_factory=list)

class SocialComment(BaseModel):
    comment_id: str
    author: AuthorInfo
    text: str
    timestamp: Optional[str] = None
    parent_id: Optional[str] = None
    replies: List['SocialComment'] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UniversalSocialPost(BaseModel):
    post_id: str
    platform: Union[Platform, str]
    post_type: PostType = PostType.text
    source_url: Optional[str] = None
    author: AuthorInfo = Field(default_factory=AuthorInfo)
    content: PostContent = Field(default_factory=lambda: PostContent(text=""))
    media: PostMedia = Field(default_factory=PostMedia)
    comments: List[SocialComment] = Field(default_factory=list)
    replies: List[SocialComment] = Field(default_factory=list)
    quoted_content: Optional['UniversalSocialPost'] = None
    reposted_content: Optional['UniversalSocialPost'] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    acquisition: AcquisitionMethod = AcquisitionMethod.unknown

    @field_validator("author", mode="before")
    @classmethod
    def parse_author(cls, v: Any) -> Any:
        if isinstance(v, str):
            return AuthorInfo(username=v, display_name=v)
        return v

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Normalize platform if string matching enum case-insensitively
        if "platform" in data and isinstance(data["platform"], str):
            p_lower = data["platform"].lower()
            allowed_enums = {p.value: p for p in Platform}
            if p_lower in allowed_enums:
                data["platform"] = allowed_enums[p_lower]

        # 1. Content backwards compatibility
        if "content" not in data or data["content"] is None:
            text = data.pop("text", "")
            title = data.pop("title", None)
            hashtags = data.pop("hashtags", [])
            mentions = data.pop("mentions", [])
            urls = data.pop("urls", [])
            data["content"] = PostContent(
                text=text or "",
                title=title,
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
            )
        elif isinstance(data["content"], dict):
            if "text" in data and "text" not in data["content"]:
                data["content"]["text"] = data.pop("text")
            if "title" in data and "title" not in data["content"]:
                data["content"]["title"] = data.pop("title")

        # 2. Media backwards compatibility
        if "media" not in data or data["media"] is None:
            images = data.pop("images", [])
            videos = data.pop("videos", [])
            audio = data.pop("audio", [])
            data["media"] = PostMedia(images=images, videos=videos, audio=audio)
        elif isinstance(data["media"], dict) and "images" in data:
            data["media"]["images"] = data.pop("images")

        # 3. PostType backwards compatibility / default
        if "post_type" not in data or data["post_type"] is None:
            images_list = []
            if isinstance(data.get("media"), PostMedia):
                images_list = data["media"].images
            elif isinstance(data.get("media"), dict):
                images_list = data["media"].get("images", [])

            text_val = ""
            if isinstance(data.get("content"), PostContent):
                text_val = data["content"].text
            elif isinstance(data.get("content"), dict):
                text_val = data["content"].get("text", "")

            if images_list and text_val:
                data["post_type"] = PostType.image_text
            elif images_list:
                data["post_type"] = PostType.image
            elif text_val:
                data["post_type"] = PostType.text
            else:
                data["post_type"] = PostType.unknown

        return data

    @computed_field
    @property
    def text(self) -> str:
        return self.content.text if self.content else ""

    @text.setter
    def text(self, value: str):
        if self.content is not None:
            self.content.text = value
        else:
            self.content = PostContent(text=value)

    @computed_field
    @property
    def title(self) -> Optional[str]:
        return self.content.title if self.content else None

    @title.setter
    def title(self, value: Optional[str]):
        if self.content is not None:
            self.content.title = value
        else:
            self.content = PostContent(text="", title=value)

    @computed_field
    @property
    def images(self) -> List[str]:
        return self.media.images if self.media else []

# Backward compatibility alias for downstream pipelines
NormalizedPost = UniversalSocialPost
NormalizedComment = SocialComment