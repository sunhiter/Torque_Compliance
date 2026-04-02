from reassemble_minexp.train.metrics import accuracy_score, binary_f1_score, classification_metrics, macro_f1_score


def test_macro_f1_score_handles_multiclass() -> None:
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]

    assert round(macro_f1_score(y_true, y_pred, num_classes=3), 4) == 0.6556


def test_binary_f1_score_handles_success_labels() -> None:
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    assert accuracy_score(y_true, y_pred) == 0.75
    assert round(binary_f1_score(y_true, y_pred), 4) == 0.6667


def test_classification_metrics_switches_by_task() -> None:
    phase_metrics = classification_metrics("phase", [0, 1, 1], [0, 0, 1], num_classes=2)
    success_metrics = classification_metrics("success", [0, 1, 1], [0, 0, 1], num_classes=2)

    assert "macro_f1" in phase_metrics
    assert "f1" not in phase_metrics
    assert "f1" in success_metrics
    assert "macro_f1" not in success_metrics
