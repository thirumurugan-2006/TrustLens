from app.preprocessing.language.code_mix_detector import CodeMixDetector
from app.preprocessing.language.language_detector import LanguageDetector
from app.preprocessing.language.script_detector import ScriptDetector
from app.preprocessing.language.transliteration_detector import TransliterationDetector


class LanguageAnalyzer:
    def __init__(self) -> None:
        self.language_detector = LanguageDetector()
        self.script_detector = ScriptDetector()
        self.code_mix_detector = CodeMixDetector()
        self.transliteration_detector = TransliterationDetector()

    def analyze(self, text: str, reliable: bool = True) -> dict:
        if not reliable or not text or not text.strip():
            return {
                "primary_language": "unknown",
                "language_confidence": 0.0,
                "secondary_languages": [],
                "scripts": [],
                "script_distribution": {},
                "is_code_mixed": False,
                "code_mix_confidence": 0.0,
                "transliteration_candidate": False,
                "transliteration_language": None,
                "transliteration_confidence": 0.0,
                "detector": "langdetect",
                "language_analysis_reliable": False,
            }

        language_result = self.language_detector.detect(text)
        script_result = self.script_detector.detect(text)
        code_mix_result = self.code_mix_detector.detect(text, language_result, script_result)
        transliteration_result = self.transliteration_detector.detect(
            text, language_result, script_result
        )

        primary_language = language_result["primary_language"]
        scripts = script_result.get("scripts", [])
        
        # Script override for high-confidence regional languages
        if "Tamil" in scripts:
            primary_language = "ta"
        elif "Devanagari" in scripts:
            primary_language = "hi"
            
        # Transliteration override
        if transliteration_result["transliteration_candidate"]:
            primary_language = transliteration_result["possible_language"]

        # If scripts show mixing but token lexicon didn't catch it
        if "Latin" in scripts and ("Tamil" in scripts or "Devanagari" in scripts):
            code_mix_result["is_code_mixed"] = True
            
        if primary_language == "en" and "ta" in code_mix_result["languages"]:
             primary_language = "ta" # prioritize regional context for trustlens

        token_languages = code_mix_result["languages"]
        secondary_languages = [
            language for language in token_languages if language != primary_language
        ]
        
        # If code mixed with Latin, English is almost always the secondary
        if code_mix_result["is_code_mixed"] and primary_language != "en" and "en" not in secondary_languages and "Latin" in scripts:
            secondary_languages.append("en")

        return {
            "primary_language": primary_language,
            "language_confidence": 0.99 if primary_language in ["hi", "ta"] else language_result["confidence"],
            "secondary_languages": secondary_languages,
            "scripts": scripts,
            "script_distribution": script_result["script_distribution"],
            "is_code_mixed": code_mix_result["is_code_mixed"],
            "code_mix_confidence": code_mix_result["confidence"],
            "transliteration_candidate": transliteration_result["transliteration_candidate"],
            "transliteration_language": transliteration_result["possible_language"],
            "transliteration_confidence": transliteration_result["confidence"],
            "detector": "ensemble",
            "language_analysis_reliable": True,
        }
