from app.preprocessing.language.code_mix_detector import token_languages


class TransliterationDetector:
    def detect(self, text: str, language_result: dict, script_result: dict) -> dict:
        scripts = script_result.get("scripts", [])
        if any(script in scripts for script in ("Tamil", "Devanagari", "Arabic")):
            return {
                "transliteration_candidate": False,
                "possible_language": None,
                "confidence": 0.0,
            }

        languages = token_languages(text)
        tamil_count = len(languages["ta"])
        hindi_count = len(languages["hi"])
        if tamil_count == 0 and hindi_count == 0:
            return {
                "transliteration_candidate": False,
                "possible_language": None,
                "confidence": 0.0,
            }

        possible_language = "ta" if tamil_count >= hindi_count else "hi"
        count = max(tamil_count, hindi_count)
        confidence = min(0.95, 0.65 + 0.08 * min(count, 3))
        return {
            "transliteration_candidate": True,
            "possible_language": possible_language,
            "confidence": round(confidence, 4),
        }
