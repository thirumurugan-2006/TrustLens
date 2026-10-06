import pytest
from app.multilingual.schemas import MultilingualRepresentation

def test_multilingual_representation_schema():
    rep = MultilingualRepresentation(
        text="Hello world",
        primary_language="en",
        languages=["en"],
        scripts=["Latin"],
        is_code_mixed=False,
        is_transliteration=False,
        model_name="xlm-roberta-base",
        embedding_dimension=768,
        embedding=[0.1] * 768,
        metadata={"test": "data"}
    )
    assert rep.text == "Hello world"
    assert rep.primary_language == "en"
    assert rep.embedding_quality == "valid"
    assert len(rep.embedding) == 768
