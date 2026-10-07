"""
TrustLens Language Feature Extractor (Phase 6D).
Extracts grounded linguistic, script, and code-mixing features from post metadata.
"""

from typing import Any, Dict, Optional

from app.training.schemas import LanguageMetadata, TrainingPost


class LanguageFeatureExtractor:
    """Extracts multilingual and script features from verified post language info."""

    LANGUAGE_MAP = {
        "en": 0,
        "ta": 1,
        "hi": 2,
        "ta-en": 3,
        "hi-en": 4,
    }

    @classmethod
    def extract(cls, post: TrainingPost) -> Dict[str, Any]:
        """Extracts language features from a TrainingPost."""
        lang_info: Optional[LanguageMetadata] = post.language_info
        if not lang_info:
            return {
                "language": "en",
                "language_id": 0,
                "script": "Latin",
                "is_code_mixed": False,
                "is_transliterated": False,
                "language_confidence": 1.0,
            }

        lang_str = (lang_info.primary or "en").lower()
        lang_id = cls.LANGUAGE_MAP.get(lang_str, 0)
        script_str = lang_info.script[0] if (lang_info.script and len(lang_info.script) > 0) else "Latin"

        # Code mixed is True if flag set or if language is ta-en / hi-en
        code_mixed = bool(lang_info.code_mixed or ("-en" in lang_str))
        transliterated = bool(lang_info.transliterated)

        return {
            "language": lang_str,
            "language_id": lang_id,
            "script": script_str,
            "is_code_mixed": code_mixed,
            "is_transliterated": transliterated,
            "language_confidence": 1.0,
        }
