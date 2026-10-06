import pytest
from app.multilingual.xlm_roberta import XLMRobertaEncoder

def test_xlm_roberta_encode():
    encoder = XLMRobertaEncoder()
    res = encoder.encode(["test"])
    
    assert len(res) == 1
    assert len(res[0]) == 768
