import os
from pathlib import Path

root = Path("d:/TrustLens")

replacements = {
    "app.input.upload.text_loader": "app.input.loaders.text_loader",
    "app.input.screenshot.screenshot_parser": "app.input.loaders.screenshot_loader",
    "app.input.reddit.reddit_adapter": "app.input.adapters.reddit_adapter",
    "app.input.router": "app.input.pipeline",
    "app.api.routes": "app.api.routes.input",
    "app.api.schemas": "app.api.schemas.responses",
    "app.vision.normalization.text_cleaner": "app.preprocessing.text.text_cleaner",
    "app.vision.normalization.pipeline": "app.preprocessing.normalization.text_normalizer",
    "app.vision.combined_ocr": "app.vision.ocr.combined_ocr",
    "app.vision.ocr_engine": "app.vision.ocr.ocr_engine",
    "app.vision.ocr_pipeline": "app.vision.ocr.ocr_pipeline",
    "app.vision.ocr_preprocessor": "app.vision.ocr.ocr_preprocessor",
    "app.vision.ocr_quality": "app.vision.ocr.ocr_quality",
    "app.vision.ocr_result": "app.vision.ocr.ocr_result",
    "app.vision.ocr_selector": "app.vision.ocr.ocr_selector",
    "app.vision.schemas": "app.vision.ocr.schemas",
    "app.vision.script_quality": "app.vision.ocr.script_quality",
    "app.vision.tamil_reader": "app.vision.ocr.tamil_reader",
    "app.vision.tamil_reliability": "app.vision.ocr.tamil_reliability",
    "app.vision.tamil_script": "app.vision.ocr.tamil_script",
}

def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated: {file_path}")

for directory in [root / "app", root / "tests", root / "scripts"]:
    for filepath in directory.rglob("*.py"):
        process_file(filepath)
