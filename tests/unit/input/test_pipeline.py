from app.pipeline.input_pipeline import InputPipeline


def test_text_pipeline():

    pipeline = InputPipeline()

    post = pipeline.process(
        "text",
        "Earn ₹50,000 per month. Pay ₹500 registration fee."
    )

    assert post.platform == "text"

    assert post.text is not None

    assert "₹50,000" in post.text

    assert "₹500" in post.text

    assert post.images == []

    assert post.comments == []


def test_pipeline_cleans_text():

    pipeline = InputPipeline()

    post = pipeline.process(
        "text",
        "Hello     world\n\nThis   is a test."
    )

    assert post.text == "Hello world This is a test."


def test_pipeline_rejects_empty_text():

    pipeline = InputPipeline()

    try:
        pipeline.process("text", "   ")
        assert False
    except ValueError:
        assert True