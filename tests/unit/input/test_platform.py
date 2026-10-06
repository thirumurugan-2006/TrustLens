from app.input.platform.detector import PlatformDetector
from app.input.schemas import Platform

def test_detector():
    detector = PlatformDetector()
    assert detector.detect("https://reddit.com/r/scams")["platform"] == Platform.reddit
    assert detector.detect("https://youtu.be/12345")["platform"] == Platform.youtube
    assert detector.detect("https://x.com/user")["platform"] == Platform.x
    assert detector.detect("https://unknown.com")["platform"] == Platform.unknown
