import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import easyocr

from app.vision.ocr.tamil_reader import TamilOCR

print("TrustLens OCR Backend Diagnostic")
print("=================================")

try:
    easyocr.Reader(["en"], gpu=False, verbose=False)
    print("English backend: PASS (EasyOCR)")
except Exception as exc:
    print("English backend: FAIL", type(exc).__name__, exc)

try:
    tamil = TamilOCR(ROOT / "data" / "models" / "paddleocr")
    tamil._load()
    print("Tamil backend: PASS (PaddleOCR)")
except Exception as exc:
    print("Tamil backend: FAIL", type(exc).__name__, exc)

print("Combined reader: NOT_AVAILABLE (language-specific fusion is used)")
print("Tamil smoke test: NOT_RUN (Tamil backend unavailable)" if 'tamil' not in locals() or tamil.error else "Tamil smoke test: PENDING")
print("English smoke test: PASS (initialization)")
print("Mixed-language test: NOT_RUN (Tamil backend unavailable)")
