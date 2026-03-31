"""Scan raw REASSEMBLE files and export a manifest."""

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
        help="Optional OmegaConf overrides, e.g. dataset.raw_root=/path/to/raw",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from reassemble_minexp.dataset.trial_index import scan_raw_files, write_manifest
    from reassemble_minexp.utils.config import load_config

    config = load_config(args.config, args.overrides)
    manifest_rows = scan_raw_files(config)
    manifest_path = write_manifest(config, manifest_rows)
    print(f"Wrote manifest with {len(manifest_rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
