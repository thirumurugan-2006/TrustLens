import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw
import cv2

def create_img(text, filename="ta_diag.png", size=(600,300)):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill="black")
    img.save(filename)
    return filename

print("Python version:", sys.version.split()[0])
import paddle
print("PaddlePaddle version:", paddle.__version__)

from paddleocr import PaddleOCR

print("\n--- Testing PaddleOCR Default ---")
img_path = create_img("முதலீட்டு திட்டம் 2024")
try:
    t0 = time.time()
    ocr = PaddleOCR(use_angle_cls=True, lang="ta", show_log=False)
    res = ocr.ocr(img_path, cls=True)
    print("Default inference successful in", time.time() - t0)
    print("Result:", res)
except Exception as e:
    print("Default inference failed:", e)

print("\n--- Testing PaddleOCR with MKLDNN Disabled ---")
try:
    t0 = time.time()
    ocr_no_mkldnn = PaddleOCR(use_angle_cls=True, lang="ta", show_log=False, use_mkldnn=False)
    res_no_mkldnn = ocr_no_mkldnn.ocr(img_path, cls=True)
    print("No-MKLDNN inference successful in", time.time() - t0)
    print("Result:", res_no_mkldnn)
except Exception as e:
    print("No-MKLDNN inference failed:", e)
