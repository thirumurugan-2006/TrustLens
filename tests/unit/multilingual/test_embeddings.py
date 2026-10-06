import pytest
from unittest.mock import patch, MagicMock
from app.multilingual.embeddings import embedding_service

@patch("app.multilingual.model_manager.MultilingualModelManager.get_model")
def test_encode_empty_text(mock_get_model):
    with pytest.raises(ValueError, match="cannot be empty"):
        embedding_service.encode("")

@patch("app.multilingual.model_manager.MultilingualModelManager.get_model")
def test_encode_valid(mock_get_model):
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]
    mock_model.embedding_dimension = 768
    mock_model.device = "cpu"
    mock_get_model.return_value = mock_model
    
    res = embedding_service.encode("test text", "xlm-roberta")
    assert res["embedding_quality"] == "valid"
    assert res["embedding_dimension"] == 768
    assert res["model_name"] == "xlm-roberta"
    assert res["device"] == "cpu"

@patch("app.multilingual.model_manager.MultilingualModelManager.get_model")
def test_encode_batch(mock_get_model):
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]
    mock_model.embedding_dimension = 768
    mock_model.device = "cpu"
    mock_get_model.return_value = mock_model
    
    res = embedding_service.encode_batch(["text1", "text2"])
    assert len(res) == 2
    assert res[0]["embedding_quality"] == "valid"
    assert res[1]["embedding_quality"] == "valid"

@patch("app.multilingual.model_manager.MultilingualModelManager.get_model")
def test_embedding_validation(mock_get_model):
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 767]  # Wrong dim
    mock_model.embedding_dimension = 768
    mock_model.device = "cpu"
    mock_get_model.return_value = mock_model
    
    res = embedding_service.encode("text")
    assert res["embedding_quality"] == "invalid"
