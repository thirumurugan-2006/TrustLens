import pytest
from app.input.adapters.base_adapter import BasePlatformAdapter, PlatformAccessError
from app.input.schemas import NormalizedPost

class DummyAdapter(BasePlatformAdapter):
    def can_handle(self, url: str) -> bool:
        return "dummy" in url
        
    def fetch(self, url: str) -> NormalizedPost:
        if not self.can_handle(url):
            raise PlatformAccessError("Invalid", platform="dummy")
        return NormalizedPost(
            post_id="1", platform="dummy", source_url=url,
            author="author", title="title", text="text",
            images=[], comments=[], timestamp=None, metadata={}
        )
        
    def health_check(self) -> dict:
        return {"platform": "dummy", "status": "available", "reason": "OK"}

def test_base_adapter():
    adapter = DummyAdapter()
    assert adapter.can_handle("http://dummy.com") is True
    assert adapter.can_handle("http://other.com") is False
    
    post = adapter.fetch("http://dummy.com")
    assert post.platform == "dummy"
    
    health = adapter.health_check()
    assert health["status"] == "available"
