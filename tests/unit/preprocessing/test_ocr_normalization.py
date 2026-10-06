import pytest
from app.preprocessing.text.text_cleaner import (
    normalize_unicode,
    remove_control_characters,
    normalize_whitespace,
    normalize_punctuation
)
from app.preprocessing.normalization.text_normalizer import _sort_regions, _remove_duplicates, normalize_ocr_result
from app.vision.ocr.ocr_result import TamilOCRResult

def test_unicode_normalization():
    # NFD to NFC
    nfd_text = "தம\u0bbfழ\u0bcd"  # Tamil text in NFD
    nfc_text = "தமிழ்"       # Tamil text in NFC
    assert normalize_unicode(nfd_text) == nfc_text

def test_whitespace_normalization():
    assert normalize_whitespace("  hello    world  ") == "hello world"
    assert normalize_whitespace("தமிழ்\n\nமொழி") == "தமிழ்\n\nமொழி"
    assert normalize_whitespace("தமிழ்\n \n \nமொழி") == "தமிழ்\n\nமொழி"

def test_punctuation_normalization():
    assert normalize_punctuation("“hello”") == "\"hello\""
    assert normalize_punctuation("‘world’") == "'world'"
    assert normalize_punctuation("COVID-19") == "COVID-19" # Must preserve hyphen
    assert normalize_punctuation("50%") == "50%"

def test_mixed_language_preservation():
    text = "இந்த vaccine நல்லது."
    res = normalize_unicode(text)
    res = normalize_whitespace(res)
    assert res == text # Should be completely unmodified

def test_numbers_and_urls():
    assert normalize_whitespace("50% of people") == "50% of people"
    assert normalize_whitespace("05-10-2026") == "05-10-2026"
    assert normalize_whitespace("https://example.com/test") == "https://example.com/test"
    assert normalize_whitespace("#தமிழ்நாடு") == "#தமிழ்நாடு"
    assert normalize_whitespace("@username") == "@username"

def test_empty_string():
    assert normalize_whitespace("") == ""
    assert normalize_whitespace("   ") == ""
    
def test_pipeline_preserves_raw():
    result = TamilOCRResult(text="  Test   String  ")
    normalized = normalize_ocr_result(result)
    assert normalized.raw_text == "  Test   String  "
    assert normalized.normalized_text == "Test String"
    assert normalized.text == "Test String"
    
def test_region_sorting():
    texts = ["bottom", "top left", "top right"]
    # [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    bboxes = [
        [[0, 100], [50, 100], [50, 120], [0, 120]],  # bottom
        [[0, 0], [50, 0], [50, 20], [0, 20]],        # top left
        [[60, 0], [100, 0], [100, 20], [60, 20]],    # top right
    ]
    confs = [0.9, 0.9, 0.9]
    
    s_texts, s_bboxes, s_confs = _sort_regions(texts, bboxes, confs)
    assert s_texts == ["top left", "top right", "bottom"]
    
def test_duplicate_removal():
    texts = ["10", "10", "This is false", "This is false", "Hello"]
    bboxes = [
        [[0,0],[10,0],[10,10],[0,10]],
        [[20,0],[30,0],[30,10],[20,10]],
        [[0,20],[50,20],[50,30],[0,30]],
        [[0,20],[50,20],[50,30],[0,30]], # Duplicate in same place
        [[0,40],[50,40],[50,50],[0,50]]
    ]
    confs = [0.9, 0.9, 0.9, 0.8, 0.9]
    
    # Current simplistic duplicate removal just checks exact text string
    # We expect 3 items: "10", "This is false", "Hello"
    u_texts, u_bboxes, u_confs = _remove_duplicates(texts, bboxes, confs)
    assert u_texts == ["10", "This is false", "Hello"]
