import re

def detect_platform(url: str) -> str:
    """
    Classify URLs into platform strings without network requests.
    """
    if not url or not isinstance(url, str):
        return "unknown"
        
    url_lower = url.lower().strip()
    
    if re.match(r"^https?://(www\.)?reddit\.com/r/[^/]+/comments/", url_lower):
        return "reddit"
        
    if re.match(r"^https?://(www\.)?linkedin\.com/posts/", url_lower):
        return "linkedin"
        
    return "unknown"
