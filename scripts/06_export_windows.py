"""Export fixed-length history windows for training and evaluation."""

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
    parser.add_argument("--config", default="configs/dataset.yaml", help="Path to dataset YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        help="Optional OmegaConf overrides, e.g. windows.history_length=32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from reassemble_minexp.dataset.window_dataset import build_window_rows, write_window_rows
    from reassemble_minexp.utils.config import load_config

    config = load_config(args.config, args.overrides)
    window_rows = build_window_rows(config)
    output_path = write_window_rows(config, window_rows)
    print(f"Wrote window export with {len(window_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
