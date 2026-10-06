import pytest
from app.multilingual.muril import MurilEncoder

def test_muril_encode():
    encoder = MurilEncoder()
    res = encoder.encode(["test"])
    
    assert len(res) == 1
    assert len(res[0]) == 768
