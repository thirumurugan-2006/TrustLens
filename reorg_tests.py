import os
import shutil
from pathlib import Path

root = Path("d:/TrustLens")
tests = root / "tests"

# Create new directory structure
new_dirs = [
    "unit/input", "unit/validation", "unit/vision", "unit/preprocessing",
    "integration", "fixtures/images", "fixtures/screenshots", "fixtures/text"
]
for d in new_dirs:
    (tests / d).mkdir(parents=True, exist_ok=True)
    (tests / d / "__init__.py").touch(exist_ok=True)

moves = {
    "test_api.py": "integration/test_api.py",
    "test_image.py": "unit/input/test_image.py",
    "test_language.py": "unit/preprocessing/test_language.py",
    "test_ocr_multilingual.py": "unit/vision/test_ocr_multilingual.py",
    "test_ocr_normalization.py": "unit/preprocessing/test_ocr_normalization.py",
    "test_pipeline.py": "unit/input/test_pipeline.py",
    "test_text.py": "unit/input/test_text.py",
    "test_validation.py": "unit/validation/test_validation.py",
    "vision/test_ocr_v2.py": "unit/vision/test_ocr_v2.py",
}

for src, dst in moves.items():
    if (tests / src).exists():
        shutil.move(tests / src, tests / dst)

# Remove old directories if empty
for d in ["vision", "benchmark"]:
    old_dir = tests / d
    if old_dir.exists():
        for pyc in old_dir.rglob("*.pyc"): pyc.unlink()
        for cdir in old_dir.rglob("__pycache__"): 
            if cdir.is_dir(): cdir.rmdir()
        if old_dir.is_dir() and not any(old_dir.iterdir()):
            old_dir.rmdir()
