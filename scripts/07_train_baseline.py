"""Train Milestone 7 baseline models on exported windows."""

from __future__ import annotations

import argparse
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
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        help="Optional OmegaConf overrides, e.g. input.modality=ft_only train.epochs=5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from omegaconf import OmegaConf

    from reassemble_minexp.train.trainer import train_baseline
    from reassemble_minexp.utils.config import load_merged_configs

    train_config = OmegaConf.load(args.config)
    dataset_config_path = train_config.get("dataset_config", "configs/dataset.yaml")
    config = load_merged_configs([dataset_config_path, args.config], args.overrides)
    metrics = train_baseline(config)
    print(
        "Finished baseline training | "
        f"task={config.task} | modality={config.input.modality} | "
        f"best_epoch={metrics['best_epoch']} | "
        f"test_metrics={metrics['test']}"
    )


if __name__ == "__main__":
    main()
