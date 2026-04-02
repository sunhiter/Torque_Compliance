"""Train Milestone 7 baseline models on exported windows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_phase.yaml", help="Path to train YAML config.")
    parser.add_argument(
        "--run-suite",
        action="store_true",
        help="Run ft_only, pose_only, and ft_pose baselines sequentially and write a summary.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        help="Optional OmegaConf overrides, e.g. input.modality=ft_only train.epochs=5",
    )
    return parser.parse_args()


def _summary_metric_key(task: str) -> str:
    return "f1" if str(task).strip().lower() == "success" else "macro_f1"


def _suite_modalities(config: object) -> list[str]:
    configured = getattr(getattr(config, "suite", object()), "modalities", None)
    if configured:
        return [str(value) for value in configured]
    return ["ft_only", "pose_only", "ft_pose"]


def _write_suite_summary(run_dir: Path, rows: list[dict[str, object]]) -> None:
    json_path = run_dir / "suite_summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    csv_path = run_dir / "suite_summary.csv"
    fieldnames = [
        "task",
        "modality",
        "model",
        "device",
        "best_epoch",
        "selection_metric",
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "train_score",
        "val_score",
        "test_score",
        "run_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    from omegaconf import OmegaConf
    from reassemble_minexp.utils.paths import ensure_directory

    from reassemble_minexp.train.trainer import train_baseline
    from reassemble_minexp.utils.config import load_merged_configs

    train_config = OmegaConf.load(args.config)
    dataset_config_path = train_config.get("dataset_config", "configs/dataset.yaml")
    base_config = load_merged_configs([dataset_config_path, args.config], args.overrides)

    if not args.run_suite:
        metrics = train_baseline(base_config)
        print(
            "Finished baseline training | "
            f"task={base_config.task} | modality={base_config.input.modality} | "
            f"best_epoch={metrics['best_epoch']} | "
            f"test_metrics={metrics['test']}"
        )
        return

    modalities = _suite_modalities(base_config)
    base_name = str(base_config.experiment.name)
    output_root = Path(str(base_config.experiment.output_root))
    suite_dir = ensure_directory(output_root / f"{base_name}_suite")
    score_key = _summary_metric_key(str(base_config.task))
    summary_rows: list[dict[str, object]] = []

    print(
        f"Running baseline suite for task={base_config.task} with modalities={modalities}",
        flush=True,
    )
    for modality in modalities:
        config = load_merged_configs([dataset_config_path, args.config], args.overrides)
        config.input.modality = modality
        config.experiment.name = f"{base_name}_{modality}"
        metrics = train_baseline(config)
        summary_rows.append(
            {
                "task": str(config.task),
                "modality": modality,
                "model": str(config.model.name),
                "device": metrics["device"],
                "best_epoch": metrics["best_epoch"],
                "selection_metric": metrics["selection_metric"],
                "train_accuracy": metrics["train"]["accuracy"],
                "val_accuracy": metrics["val"]["accuracy"],
                "test_accuracy": metrics["test"]["accuracy"],
                "train_score": metrics["train"].get(score_key, 0.0),
                "val_score": metrics["val"].get(score_key, 0.0),
                "test_score": metrics["test"].get(score_key, 0.0),
                "run_dir": metrics["run_dir"],
            }
        )

    _write_suite_summary(suite_dir, summary_rows)
    print(f"Finished baseline suite | summary_dir={suite_dir}", flush=True)
    for row in summary_rows:
        print(
            f"  {row['modality']}: val_{score_key}={row['val_score']:.4f} "
            f"test_{score_key}={row['test_score']:.4f} "
            f"run_dir={row['run_dir']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
