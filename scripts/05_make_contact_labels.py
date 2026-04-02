"""Generate weak contact labels from aligned insert samples and F/T rules."""

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
        "--suggest-thresholds",
        action="store_true",
        help="Print ft_value_norm quantiles and suggested contact thresholds after labeling.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        help="Optional OmegaConf overrides, e.g. labels.contact.touch_ft_norm_threshold=2.0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from reassemble_minexp.labels.contact_rule_labeler import (
        build_contact_labels,
        format_threshold_suggestion,
        summarize_ft_value_norms,
        write_contact_labels,
    )
    from reassemble_minexp.utils.config import load_config

    config = load_config(args.config, args.overrides)
    contact_rows = build_contact_labels(config)
    output_path = write_contact_labels(config, contact_rows)
    print(f"Wrote contact labels with {len(contact_rows)} rows to {output_path}")
    if args.suggest_thresholds:
        print()
        print(format_threshold_suggestion(summarize_ft_value_norms(contact_rows, config)))


if __name__ == "__main__":
    main()
