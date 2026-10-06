from app.benchmark.ocr_benchmark import normalize_for_comparison, sample_metrics


def test_safe_normalization_preserves_unicode():
    assert normalize_for_comparison(" தமிழ்   text ") == "தமிழ் text"


def test_exact_match_and_zero_errors():
    metrics = sample_metrics("தமிழ் text", "தமிழ் text")
    assert metrics["cer"] == 0
    assert metrics["wer"] == 0
    assert metrics["exact_match"] == 1
    assert metrics["annotation_status"] == "VALID"


def test_error_metrics_are_measured():
    metrics = sample_metrics("cat", "cut")
    assert metrics["cer"] > 0
    assert metrics["wer"] > 0
    assert metrics["exact_match"] == 0


def test_missing_annotation_is_excluded():
    metrics = sample_metrics("prediction", None)
    assert metrics["cer"] is None
    assert metrics["wer"] is None
    assert metrics["annotation_status"] == "MISSING"
