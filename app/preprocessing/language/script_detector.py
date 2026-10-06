import re
import unicodedata

_SCRIPT_RANGES = {
    "Tamil": (0x0B80, 0x0BFF),
    "Devanagari": (0x0900, 0x097F),
    "Arabic": (0x0600, 0x06FF),
}


def _script_for_character(character: str) -> str:
    codepoint = ord(character)
    if 0x0041 <= codepoint <= 0x024F:
        return "Latin"
    for script, (start, end) in _SCRIPT_RANGES.items():
        if start <= codepoint <= end:
            return script
    return "Unknown"


class ScriptDetector:
    def detect(self, text: str) -> dict:
        counts = {"Latin": 0, "Tamil": 0, "Devanagari": 0, "Arabic": 0, "Unknown": 0}
        for character in text or "":
            if not unicodedata.category(character).startswith("L"):
                continue
            script = _script_for_character(character)
            counts[script] += 1

        total = sum(counts.values())
        distribution = {
            script: round(count / total, 4) if total else 0.0
            for script, count in counts.items()
        }
        scripts = [script for script, count in counts.items() if count]
        return {
            "scripts": scripts or ["Unknown"],
            "script_distribution": distribution,
        }
