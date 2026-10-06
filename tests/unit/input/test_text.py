from app.input.loaders.text_loader import TextLoader


def test_text_loader():
    loader = TextLoader()

    post = loader.load(
        "Earn ₹50,000 per month. Pay ₹500 registration fee."
    )

    assert post.platform == "text"
    assert post.text is not None
    assert "₹50,000" in post.text
    assert "₹500" in post.text
    assert post.images == []
    assert post.comments == []


def test_text_loader_cleans_whitespace():
    loader = TextLoader()

    post = loader.load(
        "Hello    world\n\nThis   is a test."
    )

    assert post.text == "Hello world This is a test."


def test_text_loader_rejects_empty_text():
    loader = TextLoader()

    try:
        loader.load("   ")
        assert False
    except ValueError:
        assert True