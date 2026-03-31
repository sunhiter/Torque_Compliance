"""Insert-segment extraction for Milestone 3."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from reassemble_minexp.utils.paths import ensure_directory


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _is_insert_segment(segment_name: str) -> bool:
    return bool(re.match(r"^\s*insert\b", segment_name.strip(), flags=re.IGNORECASE))


def _object_from_segment_name(segment_name: str) -> str:
    match = re.match(r"^\s*insert\s+(.+?)[.!]?\s*$", segment_name, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def extract_insert_segments(
    trial_rows: list[dict[str, str]],
    segment_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Extract one row per high-level insert segment."""

    trial_by_id = {row["trial_id"]: row for row in trial_rows}
    low_level_by_parent: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in segment_rows:
        if row.get("segment_level") != "low":
            continue
        key = (row["trial_id"], row["parent_segment_id"])
        low_level_by_parent.setdefault(key, []).append(row)

    insert_rows: list[dict[str, Any]] = []
    for row in segment_rows:
        if row.get("segment_level") != "high":
            continue
        segment_name = row.get("segment_name", "")
        if not _is_insert_segment(segment_name):
            continue

        trial_id = row["trial_id"]
        high_segment_id = row["segment_id"]
        trial_row = trial_by_id.get(trial_id, {})
        low_level_rows = low_level_by_parent.get((trial_id, high_segment_id), [])
        low_level_sequence = "|".join(item["segment_name"] for item in low_level_rows if item.get("segment_name"))

        insert_rows.append(
            {
                "insert_id": f"{trial_id}::insert::{high_segment_id}",
                "trial_id": trial_id,
                "high_segment_id": high_segment_id,
                "segment_name": segment_name,
                "object_name": _object_from_segment_name(segment_name),
                "start_time": row.get("start_time", ""),
                "end_time": row.get("end_time", ""),
                "num_low_level_segments": len(low_level_rows),
                "low_level_sequence": low_level_sequence,
                "trial_object_names": trial_row.get("object_names", ""),
                "trial_success_all_actions": trial_row.get("trial_success_all_actions", ""),
                "trial_success_last_action": trial_row.get("trial_success_last_action", ""),
                "h5_path": trial_row.get("h5_path", ""),
                "pose_path": trial_row.get("pose_path", ""),
            }
        )

    return insert_rows


def build_insert_index(config: Any) -> list[dict[str, Any]]:
    """Load Milestone 2 artifacts and extract insert segments."""

    processed_root = Path(config.dataset.processed_root)
    trial_index_path = processed_root / str(config.dataset.trial_index_name)
    segment_index_path = processed_root / str(config.dataset.segment_index_name)

    if not trial_index_path.exists():
        raise FileNotFoundError(f"Trial index not found: {trial_index_path}")
    if not segment_index_path.exists():
        raise FileNotFoundError(f"Segment index not found: {segment_index_path}")

    trial_rows = _read_csv(trial_index_path)
    segment_rows = _read_csv(segment_index_path)
    return extract_insert_segments(trial_rows, segment_rows)


def write_insert_index(config: Any, insert_rows: list[dict[str, Any]]) -> Path:
    """Persist the insert index CSV."""

    processed_root = ensure_directory(config.dataset.processed_root)
    insert_index_name = str(getattr(config.dataset, "insert_index_name", "insert_index.csv"))
    insert_index_path = processed_root / insert_index_name
    _write_csv(
        insert_index_path,
        insert_rows,
        [
            "insert_id",
            "trial_id",
            "high_segment_id",
            "segment_name",
            "object_name",
            "start_time",
            "end_time",
            "num_low_level_segments",
            "low_level_sequence",
            "trial_object_names",
            "trial_success_all_actions",
            "trial_success_last_action",
            "h5_path",
            "pose_path",
        ],
    )
    return insert_index_path


def run_insert_extraction(config: Any) -> Path:
    """Run Milestone 3 insert extraction end to end."""

    insert_rows = build_insert_index(config)
    return write_insert_index(config, insert_rows)
