import pytest
from app.input.adapters.platform_detector import detect_platform

def test_detect_reddit():
    assert detect_platform("https://reddit.com/r/test/comments/123/abc") == "reddit"
    assert detect_platform("http://www.reddit.com/r/test/comments/123/") == "reddit"

def test_detect_linkedin():
    assert detect_platform("https://www.linkedin.com/posts/some-post-123") == "linkedin"
    assert detect_platform("http://linkedin.com/posts/xyz") == "linkedin"

def test_detect_unknown():
    assert detect_platform("https://twitter.com/user/status/123") == "unknown"
    assert detect_platform("not a url") == "unknown"
    assert detect_platform("") == "unknown"
    assert detect_platform(None) == "unknown"
