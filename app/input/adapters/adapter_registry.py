from typing import Type, Dict, Optional, Any
from app.input.adapters.base_adapter import BasePlatformAdapter

class AdapterRegistry:
    """
    Registry for platform adapters.
    Maps platform names to initialized adapter instances.
    """
    def __init__(self):
        self._adapters: Dict[str, BasePlatformAdapter] = {}

    def register(self, platform_name: str, adapter_instance: BasePlatformAdapter):
        """Register a platform adapter instance."""
        self._adapters[platform_name] = adapter_instance

    def get_adapter(self, platform_name: str) -> Optional[BasePlatformAdapter]:
        """Retrieve an adapter by platform name."""
        return self._adapters.get(platform_name)

    def resolve(self, url: str) -> Optional[BasePlatformAdapter]:
        """
        Iterate over registered adapters to find one that can handle the URL.
        """
        for adapter in self._adapters.values():
            if adapter.can_handle(url):
                return adapter
        return None

# Global registry instance
registry = AdapterRegistry()

# Initialize and register defaults
try:
    from app.input.adapters.reddit_adapter import RedditAdapter
    registry.register("reddit", RedditAdapter())
except ImportError:
    pass

try:
    from app.input.adapters.linkedin_adapter import LinkedInAdapter
    registry.register("linkedin", LinkedInAdapter())
except ImportError:
    pass
