"""Align insert-segment modalities onto a shared time axis."""

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
        help="Optional OmegaConf overrides, e.g. alignment.target_frequency_hz=20",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from reassemble_minexp.io.timestamp_aligner import build_aligned_index, write_aligned_index
    from reassemble_minexp.utils.config import load_config

    config = load_config(args.config, args.overrides)
    aligned_rows = build_aligned_index(config)
    output_path = write_aligned_index(config, aligned_rows)
    print(f"Wrote aligned index with {len(aligned_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
