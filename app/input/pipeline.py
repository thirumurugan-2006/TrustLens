from app.input.loaders.screenshot_loader import ScreenshotParser
from app.input.loaders.text_loader import TextLoader
from app.input.adapters.adapter_registry import registry
from app.input.adapters.platform_detector import detect_platform


class InputRouter:
    def __init__(self) -> None:
        self.text_loader = TextLoader()
        self.screenshot = ScreenshotParser()

    def route(self, input_type: str, data: str):
        if input_type == "text":
            return self.text_loader.load(data)
        if input_type == "screenshot":
            return self.screenshot.parse(data)
        if input_type in ("reddit_url", "platform_url"):
            platform = detect_platform(data)
            adapter = registry.get_adapter(platform)
            if not adapter:
                # Try generic resolution
                adapter = registry.resolve(data)
                
            if adapter:
                return adapter.fetch(data)
            raise ValueError(f"No adapter available for platform: {platform}")
            
        raise ValueError(f"Unsupported input type: {input_type}")
