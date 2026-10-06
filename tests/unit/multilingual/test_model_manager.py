import pytest
from app.multilingual.model_manager import model_manager

def test_model_availability():
    models = model_manager.get_available_models()
    assert "xlm-roberta" in models
    assert "muril" in models

def test_get_invalid_model():
    with pytest.raises(ValueError, match="Unknown model"):
        model_manager.get_model("invalid-model")
