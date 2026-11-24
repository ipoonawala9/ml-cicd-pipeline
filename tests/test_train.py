from src.train import build_model, get_threshold, quality_gate, train_and_evaluate


def test_model_pipeline_has_scaler_and_classifier():
    steps = [name for name, _ in build_model().steps]
    assert steps == ["standardscaler", "logisticregression"]


def test_accuracy_is_a_valid_probability():
    _, accuracy = train_and_evaluate()
    assert 0.0 <= accuracy <= 1.0


def test_quality_gate_passes_above_threshold():
    assert quality_gate(0.95, 0.90) is True


def test_quality_gate_fails_below_threshold():
    assert quality_gate(0.85, 0.90) is False


def test_threshold_read_from_environment(monkeypatch):
    monkeypatch.setenv("ACCURACY_THRESHOLD", "0.75")
    assert get_threshold() == 0.75
