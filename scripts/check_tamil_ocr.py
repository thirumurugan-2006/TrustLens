from pathlib import Path
import sys

import torch
import easyocr
from easyocr.config import BASE_PATH, recognition_models

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data" / "models" / "easyocr" / "tamil.pth"


def checkpoint_shape():
    if not MODEL.exists():
        return None
    checkpoint = torch.load(MODEL, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {
        key: tuple(value.shape)
        for key, value in state_dict.items()
        if hasattr(value, "shape") and "Prediction" in key
    }


def char_count(language: str) -> int:
    path = Path(BASE_PATH) / "character" / f"{language}_char.txt"
    return len(path.read_text(encoding="utf-8-sig").splitlines())


print("Tamil OCR Diagnostic")
print("====================")
print("EasyOCR version:", easyocr.__version__)
print("PyTorch version:", torch.__version__)
print("Checkpoint:", MODEL)
print("Checkpoint output tensors:", checkpoint_shape())
print("Tamil registry vocabulary:", len(recognition_models["gen1"]["tamil_g1"]["characters"]))
print("Tamil inference charset:", char_count("ta"))
print("English inference charset:", char_count("en"))
print("Combined reader used: NO")

for label, languages in (("English reader", ["en"]), ("Tamil reader", ["ta"])):
    try:
        easyocr.Reader(languages, gpu=False, model_storage_directory=str(ROOT / "data" / "models" / "easyocr"), verbose=False)
        print(f"{label}: SUCCESS")
    except Exception as exc:
        print(f"{label}: FAILURE")
        print(type(exc).__name__ + ":", exc)
