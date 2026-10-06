import re

from app.preprocessing.language.lexicons import (
    ENGLISH_WORDS,
    HINDI_TRANSLITERATED_WORDS,
    TAMIL_TRANSLITERATED_WORDS,
)

_TOKEN_PATTERN = re.compile(r"(?<![@#])\b[\w']+\b", re.UNICODE)
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def token_languages(text: str) -> dict[str, set[str]]:
    cleaned_text = _URL_PATTERN.sub(" ", text or "")
    languages = {"en": set(), "ta": set(), "hi": set()}
    for raw_token in _TOKEN_PATTERN.findall(cleaned_text.lower()):
        token = raw_token.strip("'")
        if not token or token.isnumeric():
            continue
        if any(0x0B80 <= ord(character) <= 0x0BFF for character in token):
            languages["ta"].add(token)
        elif any(0x0900 <= ord(character) <= 0x097F for character in token):
            languages["hi"].add(token)
        elif token in ENGLISH_WORDS:
            languages["en"].add(token)
        elif token in TAMIL_TRANSLITERATED_WORDS:
            languages["ta"].add(token)
        elif token in HINDI_TRANSLITERATED_WORDS:
            languages["hi"].add(token)
    return languages


class CodeMixDetector:
    def detect(self, text: str, language_result: dict, script_result: dict) -> dict:
        languages_by_token = token_languages(text)
        detected_languages = [language for language, tokens in languages_by_token.items() if tokens]
        if len(detected_languages) < 2:
            scripts = script_result.get("scripts", [])
            if len([script for script in scripts if script != "Unknown"]) >= 2:
                detected_languages = ["ta" if script == "Tamil" else "hi" if script == "Devanagari" else "en" for script in scripts]
                detected_languages = list(dict.fromkeys(detected_languages))

        is_code_mixed = len(detected_languages) >= 2
        confidence = 0.0
        if is_code_mixed:
            token_count = sum(len(tokens) for tokens in languages_by_token.values())
            confidence = min(0.95, 0.6 + 0.1 * min(token_count, 3))
        return {
            "is_code_mixed": is_code_mixed,
            "languages": detected_languages,
            "confidence": round(confidence, 4),
        }
