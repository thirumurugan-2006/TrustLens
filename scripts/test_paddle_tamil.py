from pathlib import Path
import sys
import unicodedata

import paddle
from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "test_data" / "tamil"

def contains_tamil(text: str) -> bool:
    return any("\u0B80" <= char <= "\u0BFF" for char in text)

print("PaddlePaddle:", paddle.__version__, "device:", paddle.device.get_device())
ocr = PaddleOCR(
    lang="ta",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
print("PaddleOCR: PASS")
print("Tamil model: ta_PP-OCRv5_mobile_rec")

images = sorted(path for path in IMAGE_DIR.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
if not images:
    print("Tamil smoke test: FAIL")
    print("Exact failure: no genuine Tamil image exists under test_data/tamil")
    print("Tamil Unicode detected: NO")
    sys.exit(2)

for image in images:
    result = ocr.predict(str(image))
    texts = []
    for page in result or []:
        if hasattr(page, "get"):
            texts.extend(page.get("rec_texts", []))
    text = " ".join(texts)
    print("Image:", image)
    print("Detected text:", text)
    print("Tamil Unicode detected:", "YES" if contains_tamil(text) else "NO")

