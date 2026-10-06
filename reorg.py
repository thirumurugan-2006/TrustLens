import os
import shutil
from pathlib import Path
import re

def create_dirs(base, dirs):
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)
        (base / d / "__init__.py").touch(exist_ok=True)

root = Path("d:/TrustLens")
app = root / "app"

# Create new directory structure
new_dirs = [
    "api/routes", "api/schemas",
    "input/loaders", "input/adapters",
    "validation",
    "vision/ocr", "vision/image", "vision/similarity",
    "preprocessing/language", "preprocessing/normalization", "preprocessing/text",
    "multilingual", "claims", "query_synthesis", "retrieval", "evidence", "features",
    "risk_engine", "reporting", "pipeline", "utils"
]
create_dirs(app, new_dirs)
(app / "__init__.py").touch(exist_ok=True)

# 1. API
if (app / "api/routes.py").exists():
    shutil.move(app / "api/routes.py", app / "api/routes/input.py")
if (app / "api/schemas.py").exists():
    shutil.move(app / "api/schemas.py", app / "api/schemas/responses.py")

# 2. Input
if (app / "input/screenshot/screenshot_parser.py").exists():
    shutil.move(app / "input/screenshot/screenshot_parser.py", app / "input/loaders/screenshot_loader.py")
if (app / "input/upload/text_loader.py").exists():
    shutil.move(app / "input/upload/text_loader.py", app / "input/loaders/text_loader.py")
if (app / "input/reddit/reddit_adapter.py").exists():
    shutil.move(app / "input/reddit/reddit_adapter.py", app / "input/adapters/reddit_adapter.py")

if (app / "input/router.py").exists():
    shutil.move(app / "input/router.py", app / "input/pipeline.py")

# 3. Preprocessing
if (app / "vision/normalization/text_cleaner.py").exists():
    shutil.move(app / "vision/normalization/text_cleaner.py", app / "preprocessing/text/text_cleaner.py")
if (app / "vision/normalization/pipeline.py").exists():
    shutil.move(app / "vision/normalization/pipeline.py", app / "preprocessing/normalization/text_normalizer.py")

# 4. Vision
ocr_files = [
    "combined_ocr.py", "ocr_engine.py", "ocr_pipeline.py", "ocr_preprocessor.py",
    "ocr_quality.py", "ocr_result.py", "ocr_selector.py", "schemas.py",
    "script_quality.py", "tamil_reader.py", "tamil_reliability.py", "tamil_script.py"
]
for f in ocr_files:
    if (app / f"vision/{f}").exists():
        shutil.move(app / f"vision/{f}", app / f"vision/ocr/{f}")

# Clean up empty old directories safely
for d in ["input/screenshot", "input/upload", "input/reddit", "vision/normalization", "normalization"]:
    old_dir = app / d
    if old_dir.exists():
        for pyc in old_dir.rglob("*.pyc"): pyc.unlink()
        for cdir in old_dir.rglob("__pycache__"): 
            if cdir.is_dir(): cdir.rmdir()
        if old_dir.is_dir() and not any(old_dir.iterdir()):
            old_dir.rmdir()

print("File moves completed.")
