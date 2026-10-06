import re
from app.input.schemas import Platform

class PlatformDetector:
    def __init__(self):
        self.patterns = {
            r"reddit\.com": Platform.reddit,
            r"youtube\.com|youtu\.be": Platform.youtube,
            r"instagram\.com": Platform.instagram,
            r"facebook\.com": Platform.facebook,
            r"x\.com|twitter\.com": Platform.x,
            r"linkedin\.com": Platform.linkedin,
            r"tiktok\.com": Platform.tiktok
        }

    def detect(self, url: str) -> dict:
        if not url:
            return {"platform": Platform.unknown, "confidence": 0.0, "reason": "No URL provided"}
            
        for pattern, platform in self.patterns.items():
            if re.search(pattern, url, re.IGNORECASE):
                return {"platform": platform, "confidence": 0.9, "reason": "URL match"}
                
        return {"platform": Platform.unknown, "confidence": 0.0, "reason": "No match found"}
