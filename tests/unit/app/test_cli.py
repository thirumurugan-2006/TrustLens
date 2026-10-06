import pytest
from unittest.mock import patch, MagicMock
from app.cli import run_text
from app.input.schemas import NormalizedPost

@patch("builtins.input", side_effect=["Test claim for TrustLens"])
@patch("app.pipeline.claim_pipeline.TrustLensClaimPipeline")
def test_cli_text_success(mock_pipeline_class, mock_input, capsys):
    mock_pipeline = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline
    
    mock_result = MagicMock()
    mock_result.claims = []
    mock_result.atomic_claims = []
    mock_result.search_queries = []
    mock_pipeline.process.return_value = mock_result

    with patch("os.makedirs"):
        with patch("builtins.open"):
            run_text()
    
    captured = capsys.readouterr()
    assert "PIPELINE RESULT" in captured.out
    assert "Original Text:\nTest claim for TrustLens" in captured.out
    assert "Status:\nPASS" in captured.out
