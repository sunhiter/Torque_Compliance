"""Phase-label generation from aligned samples and official low-level skills."""

from __future__ import annotations

import csv
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


def _normalize_skill_name(skill_name: str) -> str:
    return skill_name.strip().lower().replace(" ", "_")


def _phase_map(config: Any) -> dict[str, str]:
    return {
        _normalize_skill_name(skill_name): str(phase_name)
        for skill_name, phase_name in config.labels.phase_map.items()
    }


def _parse_insert_id(insert_id: str) -> tuple[str, str]:
    trial_id, high_segment_id = insert_id.split("::insert::", maxsplit=1)
    return trial_id, high_segment_id


def _low_level_lookup(segment_rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in segment_rows:
        if row.get("segment_level") != "low":
            continue
        start_time = float(row["start_time"]) if row.get("start_time") else float("-inf")
        end_time = float(row["end_time"]) if row.get("end_time") else float("inf")
        key = (row["trial_id"], row["parent_segment_id"])
        lookup.setdefault(key, []).append(
            {
                "segment_id": row["segment_id"],
                "segment_name": row["segment_name"],
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    for rows in lookup.values():
        rows.sort(key=lambda item: (item["start_time"], item["end_time"]))
    return lookup


def _active_low_level(sample_time: float, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    for row in rows:
        if row["start_time"] <= sample_time <= row["end_time"]:
            return row
    return min(
        rows,
        key=lambda row: min(abs(sample_time - row["start_time"]), abs(sample_time - row["end_time"])),
    )


def map_phase_labels(aligned_rows: list[dict[str, str]], segment_rows: list[dict[str, str]], config: Any) -> list[dict[str, Any]]:
    """Map aligned samples to phase labels using active low-level skills."""

    phase_map = _phase_map(config)
    low_level_by_insert = _low_level_lookup(segment_rows)
    phase_rows: list[dict[str, Any]] = []

    for row in aligned_rows:
        trial_id, high_segment_id = _parse_insert_id(row["insert_id"])
        low_level_rows = low_level_by_insert.get((trial_id, high_segment_id), [])
        sample_time = float(row["aligned_time"])
        active_skill = _active_low_level(sample_time, low_level_rows)
        active_skill_name = active_skill["segment_name"] if active_skill else ""
        normalized_skill = _normalize_skill_name(active_skill_name) if active_skill_name else ""
        phase_label = phase_map.get(normalized_skill, normalized_skill or "unknown")

        phase_rows.append(
            {
                "sample_id": row["sample_id"],
                "insert_id": row["insert_id"],
                "trial_id": row["trial_id"],
                "aligned_time": row["aligned_time"],
                "active_low_level_segment_id": active_skill["segment_id"] if active_skill else "",
                "active_low_level_skill": active_skill_name,
                "y_phase": phase_label,
            }
        )

    return phase_rows


def build_phase_labels(config: Any) -> list[dict[str, Any]]:
    """Load aligned samples and segment index, then generate phase labels."""

    processed_root = Path(config.dataset.processed_root)
    aligned_path = processed_root / str(config.alignment.aligned_index_name)
    segment_index_path = processed_root / str(config.dataset.segment_index_name)

    if not aligned_path.exists():
        raise FileNotFoundError(f"Aligned sample index not found: {aligned_path}")
    if not segment_index_path.exists():
        raise FileNotFoundError(f"Segment index not found: {segment_index_path}")

    aligned_rows = _read_csv(aligned_path)
    segment_rows = _read_csv(segment_index_path)
    return map_phase_labels(aligned_rows, segment_rows, config)


def write_phase_labels(config: Any, phase_rows: list[dict[str, Any]]) -> Path:
    """Persist phase labels."""

    processed_root = ensure_directory(config.dataset.processed_root)
    output_path = processed_root / str(config.labels.phase_index_name)
    _write_csv(
        output_path,
        phase_rows,
        [
            "sample_id",
            "insert_id",
            "trial_id",
            "aligned_time",
            "active_low_level_segment_id",
            "active_low_level_skill",
            "y_phase",
        ],
    )
    return output_path


def run_phase_label_pipeline(config: Any) -> Path:
    """Run Milestone 5 phase labeling end to end."""

    phase_rows = build_phase_labels(config)
    return write_phase_labels(config, phase_rows)
