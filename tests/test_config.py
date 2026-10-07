import os
from app.core.config import Settings
import pytest
import torch

def test_config_loads_defaults():
    # Unset environment to check defaults
    os.environ.clear()
    settings = Settings()
    
    assert settings.app_env == "development"
    assert settings.device == "auto"
    assert settings.retrieval_top_k == 10
    assert settings.ocr_model.name == "easyocr"

def test_device_fallback():
    os.environ["DEVICE"] = "invalid_device"
    settings = Settings()
    
    # We test the device function logic directly
    from app.core.device import get_device
    device = get_device()
    assert device in ["cpu", "cuda"]

def test_model_configuration():
    os.environ["LANGUAGE_MODEL"] = "google/muril-base-cased"
    settings = Settings()
    
    assert settings.language_model.name == "google/muril-base-cased"
    assert settings.language_model.enabled is True
