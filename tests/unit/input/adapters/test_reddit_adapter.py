import pytest
from unittest.mock import patch
from app.input.adapters.reddit_adapter import RedditAdapter, RedditAccessError

def test_reddit_can_handle():
    adapter = RedditAdapter()
    assert adapter.can_handle("https://www.reddit.com/r/test/comments/123/abc") is True
    assert adapter.can_handle("https://linkedin.com") is False

@patch("app.input.adapters.reddit_adapter.praw")
def test_reddit_health_check_unavailable(mock_praw):
    # If credentials are not provided or praw fails
    with patch.dict("os.environ", {"REDDIT_CLIENT_ID": "", "REDDIT_CLIENT_SECRET": ""}):
        adapter = RedditAdapter()
        health = adapter.health_check()
        assert health["platform"] == "reddit"
        assert health["status"] == "unavailable"

@patch("app.input.adapters.reddit_adapter.RedditAdapter.is_available", return_value=False)
def test_reddit_fetch_unavailable(mock_is_available):
    adapter = RedditAdapter()
    with pytest.raises(RedditAccessError) as exc:
        adapter.fetch("https://www.reddit.com/r/test/comments/123/abc")
    
    assert exc.value.platform == "reddit"
    assert exc.value.status == "unavailable"
