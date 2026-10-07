"""
TrustLens Text Feature Extractor (Phase 6D).
Extracts grounded structural text and lexical statistics from post content.
"""

import re
from typing import Any, Dict


class TextFeatureExtractor:
    """Extracts text structural and lexical features from post content."""

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    PHONE_PATTERN = re.compile(r"(?:\+91[\-\s]?)?[6789]\d{9}|\b\d{10}\b")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    CURRENCY_PATTERN = re.compile(r"[₹$€£]|(?:\b(?:rs|inr|usd|eur)\.?\s*\d+)", re.IGNORECASE)

    @classmethod
    def extract(cls, text: str) -> Dict[str, Any]:
        """Computes text features from raw string."""
        if not text:
            return {
                "text_length": 0,
                "word_count": 0,
                "sentence_count": 0,
                "uppercase_ratio": 0.0,
                "digit_ratio": 0.0,
                "punctuation_ratio": 0.0,
                "url_count_in_text": 0,
                "phone_count": 0,
                "email_count": 0,
                "currency_symbol_count": 0,
                "exclamation_count": 0,
                "question_count": 0,
            }

        text_len = len(text)
        words = text.split()
        word_count = len(words)

        # Sentence count (heuristics based on punctuation)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)

        # Character ratios
        alpha_chars = [c for c in text if c.isalpha()]
        uppercase_chars = [c for c in alpha_chars if c.isupper()]
        uppercase_ratio = round(len(uppercase_chars) / len(alpha_chars), 4) if alpha_chars else 0.0

        digits = [c for c in text if c.isdigit()]
        digit_ratio = round(len(digits) / text_len, 4) if text_len > 0 else 0.0

        punctuations = [c for c in text if c in "!?,.:;\"'()[]{}<>-/\\_@#$%^&*~`"]
        punctuation_ratio = round(len(punctuations) / text_len, 4) if text_len > 0 else 0.0

        # Pattern matches
        url_matches = cls.URL_PATTERN.findall(text)
        phone_matches = cls.PHONE_PATTERN.findall(text)
        email_matches = cls.EMAIL_PATTERN.findall(text)
        currency_matches = cls.CURRENCY_PATTERN.findall(text)
        exclamation_count = text.count("!")
        question_count = text.count("?")

        return {
            "text_length": text_len,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "uppercase_ratio": uppercase_ratio,
            "digit_ratio": digit_ratio,
            "punctuation_ratio": punctuation_ratio,
            "url_count_in_text": len(url_matches),
            "phone_count": len(phone_matches),
            "email_count": len(email_matches),
            "currency_symbol_count": len(currency_matches),
            "exclamation_count": exclamation_count,
            "question_count": question_count,
        }
