"""Extract insert-only high-level segments into an insert index."""

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
        help="Optional OmegaConf overrides, e.g. dataset.processed_root=data/processed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from reassemble_minexp.dataset.insert_index import build_insert_index, write_insert_index
    from reassemble_minexp.utils.config import load_config

    config = load_config(args.config, args.overrides)
    insert_rows = build_insert_index(config)
    output_path = write_insert_index(config, insert_rows)
    print(f"Wrote insert index with {len(insert_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
