import re
from typing import Dict, Any

from app.input.schemas import NormalizedPost
from app.input.adapters.base_adapter import BasePlatformAdapter, PlatformAccessError

class LinkedInAccessError(PlatformAccessError):
    """Raised when LinkedIn access is unavailable or unauthorized."""
    def __init__(self, message: str, status: str = "unavailable"):
        super().__init__(message, platform="linkedin", status=status)

class LinkedInAdapter(BasePlatformAdapter):
    """
    LinkedIn input adapter for TrustLens.
    
    Currently acts as a stub because official LinkedIn API access
    is not configured. Returns a controlled unavailable status.
    Does NOT scrape LinkedIn pages.
    """

    def __init__(self):
        # We don't have official API keys implemented right now
        self._is_configured = False

    def can_handle(self, url: str) -> bool:
        """Check whether the URL looks like a LinkedIn post URL."""
        return bool(
            re.match(
                r"^https?://(www\.)?linkedin\.com/posts/",
                url,
                re.IGNORECASE,
            )
        )

    def fetch(self, url: str) -> NormalizedPost:
        """
        Since API is not configured, always raises an error.
        Downstream code should catch PlatformAccessError.
        """
        if not self.can_handle(url):
            raise ValueError("Invalid LinkedIn post URL.")
            
        raise LinkedInAccessError(
            "LinkedIn API/access credentials are not configured",
            status="unavailable"
        )

    def health_check(self) -> Dict[str, Any]:
        """Return the health/access status of the adapter."""
        if self._is_configured:
            return {"platform": "linkedin", "status": "available", "reason": "API configured"}
        return {"platform": "linkedin", "status": "unavailable", "reason": "LinkedIn API/access credentials are not configured"}
