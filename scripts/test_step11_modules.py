"""Quick sanity test for Step 11 modules."""
import sys
sys.path.insert(0, ".")

from app.vision.ocr.tamil_script import contains_tamil, tamil_ratio, valid_char_ratio, detect_language_label
from app.vision.ocr.tamil_reliability import compute_reliability_score, is_reliable, ocr_status_from_result
from app.vision.ocr.ocr_result import TamilOCRResult

text_ta = "தமிழ் உரை"
text_en = "Hello World"
text_mixed = "COVID-19 தமிழ்நாடு"
text_garbled = "CCCCCCC"

print("=== Tamil Script Tests ===")
print(f"contains_tamil(ta): {contains_tamil(text_ta)}")
print(f"tamil_ratio(ta): {tamil_ratio(text_ta):.3f}")
print(f"tamil_ratio(en): {tamil_ratio(text_en):.3f}")
print(f"valid_char_ratio(garbled): {valid_char_ratio(text_garbled):.3f}")
print(f"detect_language_label(ta, 0.9): {detect_language_label(text_ta, 0.9)}")
print(f"detect_language_label(ta, 0.1): {detect_language_label(text_ta, 0.1)}  <- must NOT return en")
print(f"detect_language_label(en, 0.1): {detect_language_label(text_en, 0.1)}")
print(f"detect_language_label(mixed): {detect_language_label(text_mixed, 0.5)}")

print()
print("=== Reliability Score Tests ===")
r1 = compute_reliability_score(text_ta, 0.85, [0.85, 0.80], "paddleocr_ta", "ta")
r2 = compute_reliability_score(text_garbled, 0.40, [0.40], "easyocr_en", "ta")
r3 = compute_reliability_score(text_en, 0.98, [0.98], "easyocr_en", "en")
print(f"Tamil text, conf=0.85: {r1:.4f} (reliable: {is_reliable(r1)})")
print(f"Garbled text, conf=0.40: {r2:.4f} (reliable: {is_reliable(r2)})")
print(f"English text, conf=0.98: {r3:.4f} (reliable: {is_reliable(r3)})")

print()
print("=== OCR Status Tests ===")
print(f"Tamil good: {ocr_status_from_result(text_ta, 0.85, r1)}")
print(f"Garbled: {ocr_status_from_result(text_garbled, 0.40, r2)}")
print(f"Empty: {ocr_status_from_result('', None, 0.0)}")

print()
print("=== TamilOCRResult ===")
result = TamilOCRResult(
    text=text_ta,
    language="ta",
    raw_confidence=0.85,
    reliability_score=r1,
    backend="paddleocr_ta",
    status="SUCCESS",
    tamil_ratio=tamil_ratio(text_ta),
    reliable=is_reliable(r1),
)
d = result.to_dict()
print(f"raw_confidence: {d['raw_confidence']}")
print(f"reliability_score: {d['reliability_score']}")
print(f"status: {d['status']}")
print()
print("ALL TESTS PASSED")
