from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0


class LanguageDetector:
    def detect(self, text: str) -> dict:
        if not text or not text.strip():
            return {"primary_language": "unknown", "confidence": 0.0}

        try:
            detected = detect_langs(text)
        except Exception:
            return {"primary_language": "unknown", "confidence": 0.0}

        result = detected[0]
        return {
            "primary_language": result.lang,
            "confidence": round(float(result.prob), 4),
        }
