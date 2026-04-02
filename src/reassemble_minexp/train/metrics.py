"""Training metrics for Milestone 7 baselines."""

from __future__ import annotations

from typing import Iterable


def accuracy_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    truths = list(y_true)
    preds = list(y_pred)
    if not truths:
        return 0.0
    correct = sum(int(truth == pred) for truth, pred in zip(truths, preds))
    return correct / len(truths)


def macro_f1_score(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> float:
    truths = list(y_true)
    preds = list(y_pred)
    if not truths or num_classes <= 0:
        return 0.0

    scores: list[float] = []
    for label in range(num_classes):
        true_positive = sum(int(truth == label and pred == label) for truth, pred in zip(truths, preds))
        false_positive = sum(int(truth != label and pred == label) for truth, pred in zip(truths, preds))
        false_negative = sum(int(truth == label and pred != label) for truth, pred in zip(truths, preds))

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))

    return sum(scores) / len(scores)


def binary_f1_score(y_true: Iterable[int], y_pred: Iterable[int], positive_label: int = 1) -> float:
    truths = list(y_true)
    preds = list(y_pred)
    if not truths:
        return 0.0

    true_positive = sum(int(truth == positive_label and pred == positive_label) for truth, pred in zip(truths, preds))
    false_positive = sum(int(truth != positive_label and pred == positive_label) for truth, pred in zip(truths, preds))
    false_negative = sum(int(truth == positive_label and pred != positive_label) for truth, pred in zip(truths, preds))

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def classification_metrics(task: str, y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> dict[str, float]:
    truths = list(y_true)
    preds = list(y_pred)
    metrics = {"accuracy": accuracy_score(truths, preds)}
    if task == "success":
        metrics["f1"] = binary_f1_score(truths, preds)
    else:
        metrics["macro_f1"] = macro_f1_score(truths, preds, num_classes)
    return metrics
