"""Generate phase labels from aligned insert samples and low-level skills."""

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
        help="Optional OmegaConf overrides, e.g. labels.phase_index_name=phase_labels.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from reassemble_minexp.labels.phase_mapper import build_phase_labels, write_phase_labels
    from reassemble_minexp.utils.config import load_config

    config = load_config(args.config, args.overrides)
    phase_rows = build_phase_labels(config)
    output_path = write_phase_labels(config, phase_rows)
    print(f"Wrote phase labels with {len(phase_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
