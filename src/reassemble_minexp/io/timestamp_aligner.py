"""Timestamp alignment utilities for Milestone 4."""

from __future__ import annotations

import bisect
import csv
from pathlib import Path
from typing import Any

import h5py

from reassemble_minexp.utils.paths import ensure_directory


def build_uniform_timeline(start_time: float, end_time: float, target_frequency_hz: float) -> list[float]:
    """Create a uniform time axis that includes both endpoints when possible."""

    if target_frequency_hz <= 0:
        raise ValueError("target_frequency_hz must be positive")
    if end_time < start_time:
        raise ValueError("end_time must be greater than or equal to start_time")
    if start_time == end_time:
        return [start_time]

    step = 1.0 / target_frequency_hz
    timeline: list[float] = []
    current = start_time
    epsilon = step * 1e-6
    while current <= end_time + epsilon:
        timeline.append(min(current, end_time))
        current += step
    if timeline[-1] != end_time:
        timeline.append(end_time)
    return timeline


def align_nearest_indices(source_times: list[float], target_times: list[float]) -> list[int]:
    """Map each target timestamp to the nearest source index."""

    if not source_times:
        raise ValueError("source_times must not be empty")

    indices: list[int] = []
    for target in target_times:
        insert_at = bisect.bisect_left(source_times, target)
        if insert_at <= 0:
            indices.append(0)
            continue
        if insert_at >= len(source_times):
            indices.append(len(source_times) - 1)
            continue

        prev_index = insert_at - 1
        next_index = insert_at
        prev_distance = abs(target - source_times[prev_index])
        next_distance = abs(source_times[next_index] - target)
        indices.append(prev_index if prev_distance <= next_distance else next_index)
    return indices


def align_linear_indices(source_times: list[float], target_times: list[float]) -> list[dict[str, Any]]:
    """Return interpolation indices and weights for each target timestamp."""

    if not source_times:
        raise ValueError("source_times must not be empty")

    aligned: list[dict[str, Any]] = []
    for target in target_times:
        insert_at = bisect.bisect_left(source_times, target)
        if insert_at <= 0:
            aligned.append({"index_0": 0, "index_1": 0, "weight_0": 1.0, "weight_1": 0.0})
            continue
        if insert_at >= len(source_times):
            last = len(source_times) - 1
            aligned.append({"index_0": last, "index_1": last, "weight_0": 1.0, "weight_1": 0.0})
            continue

        left_index = insert_at - 1
        right_index = insert_at
        left_time = source_times[left_index]
        right_time = source_times[right_index]
        if right_time == left_time:
            aligned.append({"index_0": left_index, "index_1": right_index, "weight_0": 1.0, "weight_1": 0.0})
            continue

        weight_1 = (target - left_time) / (right_time - left_time)
        weight_0 = 1.0 - weight_1
        aligned.append(
            {
                "index_0": left_index,
                "index_1": right_index,
                "weight_0": weight_0,
                "weight_1": weight_1,
            }
        )
    return aligned


def _collect_timestamp_arrays(group: h5py.Group, prefix: str = "") -> dict[str, list[float]]:
    arrays: dict[str, list[float]] = {}
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Group):
            arrays.update(_collect_timestamp_arrays(value, path))
            continue

        raw = value[()]
        values = raw.tolist() if hasattr(raw, "tolist") else raw
        if not isinstance(values, list):
            values = [values]

        flattened: list[float] = []
        for item in values:
            if isinstance(item, list):
                if item:
                    flattened.append(float(item[0]))
            else:
                flattened.append(float(item))
        arrays[path] = flattened
    return arrays


def _match_timestamp_series(timestamp_arrays: dict[str, list[float]], candidate_keys: list[str]) -> tuple[str, list[float]]:
    for key in candidate_keys:
        if key in timestamp_arrays and timestamp_arrays[key]:
            return key, timestamp_arrays[key]
    for path, values in timestamp_arrays.items():
        if path.rsplit("/", maxsplit=1)[-1] in candidate_keys and values:
            return path, values
    return "", []


def read_alignment_sources(
    h5_path: str | Path,
    pose_keys: list[str],
    ft_keys: list[str],
    rgb_keys: list[str],
) -> dict[str, Any]:
    """Read timestamp-only alignment sources from one trial."""

    path = Path(h5_path)
    with h5py.File(path, "r") as handle:
        if "timestamps" not in handle or not isinstance(handle["timestamps"], h5py.Group):
            raise KeyError(f"No timestamps group found in {path}")
        timestamp_arrays = _collect_timestamp_arrays(handle["timestamps"])

    pose_key, pose_times = _match_timestamp_series(timestamp_arrays, pose_keys)
    ft_key, ft_times = _match_timestamp_series(timestamp_arrays, ft_keys)
    rgb_streams = {key: _match_timestamp_series(timestamp_arrays, [key])[1] for key in rgb_keys}

    return {
        "pose_key": pose_key,
        "pose_times": pose_times,
        "ft_key": ft_key,
        "ft_times": ft_times,
        "rgb_streams": rgb_streams,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _alignment_records(source_times: list[float], target_times: list[float], mode: str) -> list[dict[str, Any]]:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "nearest":
        nearest = align_nearest_indices(source_times, target_times)
        return [
            {"index_0": index, "index_1": index, "weight_0": 1.0, "weight_1": 0.0}
            for index in nearest
        ]
    if normalized_mode == "linear":
        return align_linear_indices(source_times, target_times)
    raise ValueError(f"Unsupported alignment mode: {mode}")


def align_insert_rows(insert_rows: list[dict[str, str]], config: Any) -> list[dict[str, Any]]:
    """Build aligned sample-index rows for all insert segments."""

    target_frequency_hz = float(config.alignment.target_frequency_hz)
    pose_mode = str(config.alignment.pose_mode)
    ft_mode = str(config.alignment.ft_mode)
    rgb_keys = list(config.alignment.rgb_source_keys)
    pose_keys = list(config.alignment.pose_source_keys)
    ft_keys = list(config.alignment.ft_source_keys)

    source_cache: dict[str, dict[str, Any]] = {}
    aligned_rows: list[dict[str, Any]] = []

    for insert_row in insert_rows:
        h5_path = insert_row["h5_path"]
        if h5_path not in source_cache:
            source_cache[h5_path] = read_alignment_sources(h5_path, pose_keys, ft_keys, rgb_keys)
        sources = source_cache[h5_path]

        start_time = float(insert_row["start_time"])
        end_time = float(insert_row["end_time"])
        timeline = build_uniform_timeline(start_time, end_time, target_frequency_hz)

        pose_records = _alignment_records(sources["pose_times"], timeline, pose_mode) if sources["pose_times"] else [{} for _ in timeline]
        ft_records = _alignment_records(sources["ft_times"], timeline, ft_mode) if sources["ft_times"] else [{} for _ in timeline]
        rgb_indices = {
            key: align_nearest_indices(times, timeline) if times else []
            for key, times in sources["rgb_streams"].items()
        }

        for sample_index, aligned_time in enumerate(timeline):
            pose_record = pose_records[sample_index]
            ft_record = ft_records[sample_index]
            row = {
                "sample_id": f"{insert_row['insert_id']}::sample::{sample_index:05d}",
                "insert_id": insert_row["insert_id"],
                "trial_id": insert_row["trial_id"],
                "sample_index": sample_index,
                "aligned_time": aligned_time,
                "pose_source_key": sources["pose_key"],
                "pose_index_0": pose_record.get("index_0", ""),
                "pose_index_1": pose_record.get("index_1", ""),
                "pose_weight_0": pose_record.get("weight_0", ""),
                "pose_weight_1": pose_record.get("weight_1", ""),
                "ft_source_key": sources["ft_key"],
                "ft_index_0": ft_record.get("index_0", ""),
                "ft_index_1": ft_record.get("index_1", ""),
                "ft_weight_0": ft_record.get("weight_0", ""),
                "ft_weight_1": ft_record.get("weight_1", ""),
                "h5_path": insert_row["h5_path"],
                "pose_path": insert_row["pose_path"],
            }
            for rgb_key in rgb_keys:
                indices = rgb_indices.get(rgb_key, [])
                row[f"rgb_{rgb_key}_index"] = indices[sample_index] if sample_index < len(indices) else ""
            aligned_rows.append(row)

    return aligned_rows


def build_aligned_index(config: Any) -> list[dict[str, Any]]:
    """Load insert segments and build aligned sample rows."""

    processed_root = Path(config.dataset.processed_root)
    insert_index_name = str(getattr(config.dataset, "insert_index_name", "insert_index.csv"))
    insert_index_path = processed_root / insert_index_name
    if not insert_index_path.exists():
        raise FileNotFoundError(f"Insert index not found: {insert_index_path}")

    insert_rows = _read_csv(insert_index_path)
    return align_insert_rows(insert_rows, config)


def write_aligned_index(config: Any, aligned_rows: list[dict[str, Any]]) -> Path:
    """Persist aligned sample-index rows."""

    processed_root = ensure_directory(config.dataset.processed_root)
    output_path = processed_root / str(config.alignment.aligned_index_name)
    fieldnames = [
        "sample_id",
        "insert_id",
        "trial_id",
        "sample_index",
        "aligned_time",
        "pose_source_key",
        "pose_index_0",
        "pose_index_1",
        "pose_weight_0",
        "pose_weight_1",
        "ft_source_key",
        "ft_index_0",
        "ft_index_1",
        "ft_weight_0",
        "ft_weight_1",
    ]
    fieldnames.extend(f"rgb_{rgb_key}_index" for rgb_key in config.alignment.rgb_source_keys)
    fieldnames.extend(["h5_path", "pose_path"])
    _write_csv(output_path, aligned_rows, fieldnames)
    return output_path


def run_alignment_pipeline(config: Any) -> Path:
    """Run Milestone 4 alignment end to end."""

    aligned_rows = build_aligned_index(config)
    return write_aligned_index(config, aligned_rows)
