from app.preprocessing.language.language_analyzer import LanguageAnalyzer


analyzer = LanguageAnalyzer()


def test_english_text():
    result = analyzer.analyze("This is a normal English sentence.")
    assert result["primary_language"] == "en"
    assert result["is_code_mixed"] is False
    assert result["scripts"] == ["Latin"]


def test_tamil_text():
    result = analyzer.analyze("வணக்கம் இது ஒரு தமிழ் பதிவு")
    assert result["primary_language"] == "ta"
    assert "Tamil" in result["scripts"]


def test_tamil_english_mixed_text():
    result = analyzer.analyze("Hello நண்பர்களே")
    assert result["is_code_mixed"] is True
    assert "Latin" in result["scripts"]
    assert "Tamil" in result["scripts"]


def test_tanglish_transliteration_candidate():
    result = analyzer.analyze("naan oru job theduren")
    assert result["scripts"] == ["Latin"]
    assert result["transliteration_candidate"] is True
    assert result["transliteration_language"] == "ta"


def test_hinglish_transliteration_candidate():
    result = analyzer.analyze("mera job ready hai")
    assert result["transliteration_candidate"] is True
    assert result["transliteration_language"] == "hi"


def test_currency_and_numbers_do_not_change_script():
    result = analyzer.analyze("Earn ₹500 every month")
    assert result["primary_language"] == "en"
    assert result["scripts"] == ["Latin"]


def test_url_does_not_dominate_language_detection():
    result = analyzer.analyze("https://example.com job offer")
    assert result["primary_language"] == "en"
    assert result["scripts"] == ["Latin"]


def test_empty_text():
    result = analyzer.analyze("")
    assert result["primary_language"] == "unknown"
    assert result["language_confidence"] == 0.0
