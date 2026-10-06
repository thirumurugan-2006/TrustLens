from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
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
    platform: Platform
    post_type: PostType
    source_url: Optional[str] = None
    author: AuthorInfo = Field(default_factory=AuthorInfo)
    content: PostContent
    media: PostMedia = Field(default_factory=PostMedia)
    comments: List[SocialComment] = Field(default_factory=list)
    replies: List[SocialComment] = Field(default_factory=list)
    quoted_content: Optional['UniversalSocialPost'] = None
    reposted_content: Optional['UniversalSocialPost'] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    acquisition: AcquisitionMethod = AcquisitionMethod.unknown