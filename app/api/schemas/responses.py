from pydantic import BaseModel


class TextInputRequest(BaseModel):
    text: str


class RedditInputRequest(BaseModel):
    url: str

class PlatformInputRequest(BaseModel):
    url: str
