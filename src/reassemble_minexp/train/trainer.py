"""Baseline training loop for Milestone 7."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from reassemble_minexp.models.gru import GRUClassifier
from reassemble_minexp.models.mlp import MLPClassifier
from reassemble_minexp.train.losses import build_classification_loss
from reassemble_minexp.train.metrics import classification_metrics
from reassemble_minexp.utils.logging_utils import prepare_run_directory, write_json, write_resolved_config
from reassemble_minexp.utils.seed import set_global_seed


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_success_label(value: str) -> str:
    return "True" if str(value).strip().lower() in {"1", "true", "yes", "success"} else "False"


def _task_target_key(task: str) -> str:
    if task == "phase":
        return "y_phase"
    if task == "contact":
        return "y_contact"
    if task == "success":
        return "y_success"
    raise ValueError(f"Unsupported task: {task}")


def _modality_flags(modality: str) -> tuple[bool, bool]:
    normalized = modality.strip().lower()
    if normalized == "ft_only":
        return True, False
    if normalized == "pose_only":
        return False, True
    if normalized in {"ft_pose", "pose_ft"}:
        return True, True
    raise ValueError(f"Unsupported modality setting: {modality}")


def _load_feature_sequence(row: dict[str, str], use_ft: bool, use_pose: bool) -> list[list[float]]:
    ft_history = json.loads(row["ft_history_json"]) if use_ft else []
    pose_history = json.loads(row["pose_history_json"]) if use_pose else []
    history_length = len(ft_history) if ft_history else len(pose_history)
    if history_length == 0:
        raise ValueError("Window row does not contain usable history features.")

    features: list[list[float]] = []
    for index in range(history_length):
        step: list[float] = []
        if ft_history:
            step.extend(float(value) for value in ft_history[index])
        if pose_history:
            step.extend(float(value) for value in pose_history[index])
        features.append(step)
    return features


def _label_value(row: dict[str, str], task: str) -> str:
    raw = row[_task_target_key(task)]
    return _normalize_success_label(raw) if task == "success" else raw


def _build_label_mapping(rows_by_split: dict[str, list[dict[str, str]]], task: str) -> dict[str, int]:
    labels = {_label_value(row, task) for rows in rows_by_split.values() for row in rows}
    if task == "success":
        ordered = [label for label in ("False", "True") if label in labels]
    else:
        ordered = sorted(labels)
    return {label: index for index, label in enumerate(ordered)}


def _selection_metric(task: str, config: Any) -> str:
    configured = str(getattr(config.train, "selection_metric", "")).strip()
    if configured:
        return configured
    return "f1" if task == "success" else "macro_f1"


def _resolve_device(config: Any) -> torch.device:
    configured = str(getattr(config.train, "device", "auto")).strip().lower()
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)


@dataclass
class WindowSample:
    features: torch.Tensor
    label: int


class WindowClassificationDataset(Dataset[WindowSample]):
    """Baseline dataset backed by exported window CSVs."""

    def __init__(self, rows: list[dict[str, str]], task: str, modality: str, label_mapping: dict[str, int]) -> None:
        use_ft, use_pose = _modality_flags(modality)
        self.samples: list[WindowSample] = []
        for row in rows:
            features = torch.tensor(_load_feature_sequence(row, use_ft, use_pose), dtype=torch.float32)
            label_name = _label_value(row, task)
            self.samples.append(WindowSample(features=features, label=label_mapping[label_name]))

        if not self.samples:
            raise ValueError("No samples available for baseline training.")

        self.sequence_length = int(self.samples[0].features.shape[0])
        self.feature_dim = int(self.samples[0].features.shape[1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        return sample.features, torch.tensor(sample.label, dtype=torch.long)


def _build_model(config: Any, feature_dim: int, sequence_length: int, num_classes: int) -> nn.Module:
    model_name = str(config.model.name).strip().lower()
    if model_name == "mlp":
        return MLPClassifier(
            input_dim=feature_dim,
            sequence_length=sequence_length,
            num_classes=num_classes,
            hidden_dims=list(getattr(config.model, "hidden_dims", [256, 128])),
            dropout=float(getattr(config.model, "dropout", 0.1)),
        )
    if model_name == "gru":
        return GRUClassifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_size=int(getattr(config.model, "hidden_size", 128)),
            num_layers=int(getattr(config.model, "num_layers", 1)),
            dropout=float(getattr(config.model, "dropout", 0.1)),
            bidirectional=bool(getattr(config.model, "bidirectional", False)),
        )
    raise ValueError(f"Unsupported baseline model: {config.model.name}")


def _compute_class_weights(dataset: WindowClassificationDataset, num_classes: int) -> list[float]:
    counts: dict[int, int] = {}
    for sample in dataset.samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    total = sum(counts.values())
    return [
        total / (num_classes * counts[index]) if counts.get(index, 0) else 1.0
        for index in range(num_classes)
    ]


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[int]]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        logits = model(features)
        loss = loss_fn(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * int(labels.shape[0])
        predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    average_loss = total_loss / len(dataloader.dataset)
    return average_loss, predictions, targets


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
    task: str,
    num_classes: int,
) -> dict[str, float]:
    with torch.no_grad():
        loss, predictions, targets = _run_epoch(model, dataloader, loss_fn, device, optimizer=None)
    metrics = classification_metrics(task, targets, predictions, num_classes)
    metrics["loss"] = loss
    return metrics


def _dataloader(dataset: WindowClassificationDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_baseline(config: Any) -> dict[str, Any]:
    """Train a minimal baseline classifier from exported windows."""

    task = str(config.task).strip().lower()
    dataset_processed_root = Path(config.dataset.processed_root)
    rows_by_split = {
        "train": _read_rows(dataset_processed_root / str(config.windows.split_output_names.train)),
        "val": _read_rows(dataset_processed_root / str(config.windows.split_output_names.val)),
        "test": _read_rows(dataset_processed_root / str(config.windows.split_output_names.test)),
    }

    label_mapping = _build_label_mapping(rows_by_split, task)
    datasets = {
        split_name: WindowClassificationDataset(rows, task, str(config.input.modality), label_mapping)
        for split_name, rows in rows_by_split.items()
    }

    set_global_seed(int(getattr(config.train, "seed", 7)))
    device = _resolve_device(config)
    train_dataset = datasets["train"]
    num_classes = len(label_mapping)
    model = _build_model(config, train_dataset.feature_dim, train_dataset.sequence_length, num_classes).to(device)

    train_loader = _dataloader(
        train_dataset,
        batch_size=int(config.train.batch_size),
        shuffle=True,
        num_workers=int(getattr(config.train, "num_workers", 0)),
    )
    val_loader = _dataloader(
        datasets["val"],
        batch_size=int(config.train.batch_size),
        shuffle=False,
        num_workers=int(getattr(config.train, "num_workers", 0)),
    )
    test_loader = _dataloader(
        datasets["test"],
        batch_size=int(config.train.batch_size),
        shuffle=False,
        num_workers=int(getattr(config.train, "num_workers", 0)),
    )

    class_weights = _compute_class_weights(train_dataset, num_classes) if bool(getattr(config.train, "use_class_weights", False)) else None
    loss_fn = build_classification_loss(class_weights).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.train.learning_rate),
        weight_decay=float(getattr(config.train, "weight_decay", 0.0)),
    )

    output_root = getattr(config.experiment, "output_root", "outputs/baselines")
    run_dir = prepare_run_directory(output_root, str(config.experiment.name))
    checkpoint_path = run_dir / str(getattr(config.train, "checkpoint_name", "best.pt"))
    selection_metric = _selection_metric(task, config)
    best_metric = float("-inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config.train.epochs) + 1):
        train_loss, train_predictions, train_targets = _run_epoch(model, train_loader, loss_fn, device, optimizer=optimizer)
        train_metrics = classification_metrics(task, train_targets, train_predictions, num_classes)
        train_metrics["loss"] = train_loss
        val_metrics = _evaluate(model, val_loader, loss_fn, device, task, num_classes)

        epoch_summary = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_summary)
        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val {selection_metric}={val_metrics.get(selection_metric, 0.0):.4f}"
        )

        score = float(val_metrics.get(selection_metric, 0.0))
        if score > best_metric:
            best_metric = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_mapping": label_mapping,
                    "config": OmegaConf.to_container(config, resolve=True),
                    "epoch": epoch,
                    "selection_metric": selection_metric,
                    "best_metric": best_metric,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = {
        "train": _evaluate(model, train_loader, loss_fn, device, task, num_classes),
        "val": _evaluate(model, val_loader, loss_fn, device, task, num_classes),
        "test": _evaluate(model, test_loader, loss_fn, device, task, num_classes),
    }
    final_metrics["best_epoch"] = int(checkpoint["epoch"])
    final_metrics["selection_metric"] = selection_metric
    final_metrics["label_mapping"] = label_mapping
    final_metrics["feature_dim"] = train_dataset.feature_dim
    final_metrics["sequence_length"] = train_dataset.sequence_length
    final_metrics["device"] = str(device)
    final_metrics["run_dir"] = str(run_dir)

    write_json(run_dir / str(getattr(config.train, "metrics_name", "metrics.json")), final_metrics)
    write_json(run_dir / str(getattr(config.train, "history_name", "history.json")), {"epochs": history})
    write_json(run_dir / str(getattr(config.train, "label_mapping_name", "label_mapping.json")), label_mapping)
    write_resolved_config(run_dir / "resolved_config.yaml", config)

    return final_metrics
