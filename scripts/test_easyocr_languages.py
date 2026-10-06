import easyocr
import torch

print(f"EasyOCR version: {easyocr.__version__}")
print(f"EasyOCR path: {easyocr.__file__}")
print(f"PyTorch version: {torch.__version__}")

for label, languages in (("English reader", ["en"]), ("Tamil reader", ["ta"])):
    print(f"\nTesting {label}: {languages}")
    try:
        easyocr.Reader(languages, gpu=False, verbose=False)
        print("SUCCESS")
    except Exception as exc:
        print("FAILURE")
        print(f"{type(exc).__name__}: {exc}")
