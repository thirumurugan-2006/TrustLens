import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Sub-directories for raw data
RAW_SCREENSHOTS_DIR = RAW_DIR / "screenshots"
RAW_REDDIT_DIR = RAW_DIR / "reddit"
RAW_DOCUMENTS_DIR = RAW_DIR / "documents"
RAW_IMAGES_DIR = RAW_DIR / "images"

# Sub-directories for processed data
NORMALIZED_DIR = PROCESSED_DIR / "normalized"
NORMALIZED_SCREENSHOTS_DIR = NORMALIZED_DIR / "screenshots"
NORMALIZED_REDDIT_DIR = NORMALIZED_DIR / "reddit"
NORMALIZED_TEXT_DIR = NORMALIZED_DIR / "text"
OCR_PROCESSED_DIR = PROCESSED_DIR / "ocr"
LANGUAGE_PROCESSED_DIR = PROCESSED_DIR / "language"
FEATURES_PROCESSED_DIR = PROCESSED_DIR / "features"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"
EASYOCR_MODELS_DIR = MODELS_DIR / "easyocr"
EMBEDDINGS_MODELS_DIR = MODELS_DIR / "embeddings"
MULTILINGUAL_MODELS_DIR = MODELS_DIR / "multilingual"

# Ensure all directories exist
for directory in [
    RAW_SCREENSHOTS_DIR, RAW_REDDIT_DIR, RAW_DOCUMENTS_DIR, RAW_IMAGES_DIR,
    NORMALIZED_SCREENSHOTS_DIR, NORMALIZED_REDDIT_DIR, NORMALIZED_TEXT_DIR,
    OCR_PROCESSED_DIR, LANGUAGE_PROCESSED_DIR, FEATURES_PROCESSED_DIR,
    EASYOCR_MODELS_DIR, EMBEDDINGS_MODELS_DIR, MULTILINGUAL_MODELS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
