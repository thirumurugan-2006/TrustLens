import pytest
from app.input.adapters.linkedin_adapter import LinkedInAdapter, LinkedInAccessError

def test_linkedin_can_handle():
    adapter = LinkedInAdapter()
    assert adapter.can_handle("https://www.linkedin.com/posts/example") is True
    assert adapter.can_handle("https://reddit.com") is False

def test_linkedin_fetch_unavailable():
    adapter = LinkedInAdapter()
    with pytest.raises(LinkedInAccessError) as exc:
        adapter.fetch("https://www.linkedin.com/posts/example")
    
    assert "not configured" in str(exc.value)
    assert exc.value.platform == "linkedin"
    assert exc.value.status == "unavailable"

def test_linkedin_health_check():
    adapter = LinkedInAdapter()
    health = adapter.health_check()
    assert health["platform"] == "linkedin"
    assert health["status"] == "unavailable"
