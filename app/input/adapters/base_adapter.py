from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from app.input.schemas import NormalizedPost

class BasePlatformAdapter(ABC):
    """
    Abstract base class for all platform adapters in TrustLens.
    """

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """
        Check whether this adapter can handle the given URL.
        """
        pass

    @abstractmethod
    def fetch(self, url: str) -> NormalizedPost:
        """
        Fetch content from the given URL and normalize it.
        Raises PlatformAccessError if unavailable.
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Return the health/access status of the adapter.
        Expected keys: 'platform', 'status', 'reason'
        """
        pass

class PlatformAccessError(Exception):
    """Raised when platform access is unavailable or fails gracefully."""
    def __init__(self, message: str, platform: str, status: str = "unavailable"):
        super().__init__(message)
        self.platform = platform
        self.status = status
