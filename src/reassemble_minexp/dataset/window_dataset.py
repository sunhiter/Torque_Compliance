"""Window export for Milestone 6."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import yaml

from reassemble_minexp.io.h5_reader import read_h5_metadata
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


def _normalize_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "success"}:
            return True
        if lowered in {"false", "0", "no", "failure"}:
            return False
    return value


def _first_present(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _segment_value(segment: dict[str, Any], keys: list[str]) -> Any:
    value = _first_present(segment, keys)
    if value is not None:
        return value
    attrs = segment.get("attrs", {})
    if isinstance(attrs, dict):
        return _first_present(attrs, keys)
    return None


def _high_level_segments(segments_info: Any) -> list[dict[str, Any]]:
    if isinstance(segments_info, list):
        return [item for item in segments_info if isinstance(item, dict)]
    if isinstance(segments_info, dict):
        if "segments" in segments_info and isinstance(segments_info["segments"], list):
            return [item for item in segments_info["segments"] if isinstance(item, dict)]
        if "high_level" in segments_info and isinstance(segments_info["high_level"], list):
            return [item for item in segments_info["high_level"] if isinstance(item, dict)]
        return [value for key, value in segments_info.items() if key != "attrs" and isinstance(value, dict)]
    return []


def _segment_success_lookup(h5_path: str, result_keys: list[str], segments_key: str) -> dict[str, Any]:
    metadata = read_h5_metadata(h5_path, segments_key)
    lookup: dict[str, Any] = {}
    for index, segment in enumerate(_high_level_segments(metadata.get("segments_info"))):
        segment_id = str(_segment_value(segment, ["id"]) or index)
        lookup[segment_id] = _normalize_bool(_segment_value(segment, result_keys))
    return lookup


def _dataset_path_candidates(source_key: str) -> list[str]:
    candidates = [source_key]
    if "/" not in source_key:
        candidates.extend([f"robot_state/{source_key}"])
    return candidates


def _read_numeric_dataset(handle: h5py.File, source_key: str) -> list[list[float]]:
    for candidate in _dataset_path_candidates(source_key):
        if candidate not in handle:
            continue
        raw = handle[candidate][()]
        values = raw.tolist() if hasattr(raw, "tolist") else raw
        if not isinstance(values, list):
            values = [values]
        rows: list[list[float]] = []
        for item in values:
            if isinstance(item, list):
                rows.append([float(value) for value in item])
            else:
                rows.append([float(item)])
        return rows
    raise KeyError(f"Could not find dataset for source key: {source_key}")


def _interpolate_vector(samples: list[list[float]], index_0: int, index_1: int, weight_0: float, weight_1: float) -> list[float]:
    left = samples[index_0]
    right = samples[index_1]
    width = min(len(left), len(right))
    return [(left[i] * weight_0) + (right[i] * weight_1) for i in range(width)]


def _parse_json_or_scalar(value: str) -> Any:
    if value in ("", None):
        return ""
    return value


def _load_structure_tokens(path: str | Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {"insert_id": {}, "trial_id": {}, "object_name": {}}

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Structure token file not found: {file_path}")

    mappings = {"insert_id": {}, "trial_id": {}, "object_name": {}}
    if file_path.suffix.lower() == ".csv":
        rows = _read_csv(file_path)
        for row in rows:
            token = row.get("structure_token", "")
            if row.get("insert_id"):
                mappings["insert_id"][row["insert_id"]] = token
            if row.get("trial_id"):
                mappings["trial_id"][row["trial_id"]] = token
            if row.get("object_name"):
                mappings["object_name"][row["object_name"]] = token
        return mappings

    payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("insert_id", "trial_id", "object_name"):
            values = payload.get(key, {})
            if isinstance(values, dict):
                mappings[key] = {str(inner_key): str(inner_value) for inner_key, inner_value in values.items()}
    return mappings


def _resolve_structure_token(insert_row: dict[str, str], token_maps: dict[str, dict[str, str]]) -> str:
    insert_id = insert_row.get("insert_id", "")
    trial_id = insert_row.get("trial_id", "")
    object_name = insert_row.get("object_name", "") or insert_row.get("trial_object_names", "")
    return (
        token_maps["insert_id"].get(insert_id)
        or token_maps["trial_id"].get(trial_id)
        or token_maps["object_name"].get(object_name)
        or object_name
    )


def _split_lookup(config: Any) -> dict[str, str]:
    split_source = getattr(config.windows, "split_source_path", "")
    if not split_source:
        return {}

    path = Path(str(split_source))
    if not path.exists():
        raise FileNotFoundError(f"Split source file not found: {path}")
    rows = _read_csv(path)
    lookup: dict[str, str] = {}
    for row in rows:
        key = row.get("insert_id") or row.get("trial_id")
        split = row.get("split", "")
        if key and split:
            lookup[key] = split
    return lookup


def _deterministic_split(trial_id: str, config: Any) -> str:
    train_ratio = float(config.windows.split_ratios.train)
    val_ratio = float(config.windows.split_ratios.val)
    digest = hashlib.md5(trial_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(16**8)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def _assign_split(insert_row: dict[str, str], split_lookup: dict[str, str], config: Any) -> str:
    insert_id = insert_row["insert_id"]
    trial_id = insert_row["trial_id"]
    return split_lookup.get(insert_id) or split_lookup.get(trial_id) or _deterministic_split(trial_id, config)


def _group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def _sorted_samples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: int(row["sample_index"]))


def _history_indices(target_index: int, history_length: int, history_step: int) -> list[int]:
    start_index = target_index - (history_length - 1) * history_step
    return [start_index + offset * history_step for offset in range(history_length)]


def build_window_rows(config: Any) -> list[dict[str, Any]]:
    """Build fixed-length history windows from aligned samples and labels."""

    processed_root = Path(config.dataset.processed_root)
    aligned_rows = _read_csv(processed_root / str(config.alignment.aligned_index_name))
    phase_rows = _read_csv(processed_root / str(config.labels.phase_index_name))
    contact_rows = _read_csv(processed_root / str(config.labels.contact_index_name))
    insert_rows = _read_csv(processed_root / str(config.dataset.insert_index_name))

    aligned_by_insert = {insert_id: _sorted_samples(rows) for insert_id, rows in _group_rows(aligned_rows, "insert_id").items()}
    phase_by_sample = {row["sample_id"]: row for row in phase_rows}
    contact_by_sample = {row["sample_id"]: row for row in contact_rows}
    insert_by_id = {row["insert_id"]: row for row in insert_rows}

    history_length = int(config.windows.history_length)
    history_step = int(config.windows.history_step)
    export_stride = int(config.windows.export_stride)
    token_maps = _load_structure_tokens(getattr(config.windows, "structure_tokens_path", ""))
    split_lookup = _split_lookup(config)

    h5_cache: dict[str, dict[str, list[list[float]]]] = {}
    success_cache: dict[str, dict[str, Any]] = {}
    window_rows: list[dict[str, Any]] = []

    for insert_id, samples in aligned_by_insert.items():
        if len(samples) < history_length or len(samples) < 2:
            continue

        insert_row = insert_by_id[insert_id]
        split = _assign_split(insert_row, split_lookup, config)
        structure_token = _resolve_structure_token(insert_row, token_maps)
        h5_path = insert_row["h5_path"]
        if h5_path not in success_cache:
            success_cache[h5_path] = _segment_success_lookup(
                h5_path,
                list(config.schema.result_keys),
                str(config.schema.segments_key),
            )
        insert_success = success_cache[h5_path].get(insert_row["high_segment_id"], insert_row.get("trial_success_last_action", ""))

        if h5_path not in h5_cache:
            h5_cache[h5_path] = {}

        for target_index in range((history_length - 1) * history_step, len(samples) - 1, export_stride):
            indices = _history_indices(target_index, history_length, history_step)
            history_samples = [samples[index] for index in indices]
            next_sample = samples[target_index + 1]
            current_sample = samples[target_index]

            pose_source_key = current_sample["pose_source_key"]
            ft_source_key = current_sample["ft_source_key"]
            with h5py.File(h5_path, "r") as handle:
                if pose_source_key and pose_source_key not in h5_cache[h5_path]:
                    h5_cache[h5_path][pose_source_key] = _read_numeric_dataset(handle, pose_source_key)
                if ft_source_key and ft_source_key not in h5_cache[h5_path]:
                    h5_cache[h5_path][ft_source_key] = _read_numeric_dataset(handle, ft_source_key)

            pose_samples = h5_cache[h5_path].get(pose_source_key, [])
            ft_samples = h5_cache[h5_path].get(ft_source_key, [])

            pose_history = []
            ft_history = []
            for sample in history_samples:
                if pose_samples:
                    pose_history.append(
                        _interpolate_vector(
                            pose_samples,
                            int(sample["pose_index_0"]),
                            int(sample["pose_index_1"]),
                            float(sample["pose_weight_0"]),
                            float(sample["pose_weight_1"]),
                        )
                    )
                if ft_samples:
                    ft_history.append(
                        _interpolate_vector(
                            ft_samples,
                            int(sample["ft_index_0"]),
                            int(sample["ft_index_1"]),
                            float(sample["ft_weight_0"]),
                            float(sample["ft_weight_1"]),
                        )
                    )

            current_pose = pose_history[-1] if pose_history else []
            next_pose = (
                _interpolate_vector(
                    pose_samples,
                    int(next_sample["pose_index_0"]),
                    int(next_sample["pose_index_1"]),
                    float(next_sample["pose_weight_0"]),
                    float(next_sample["pose_weight_1"]),
                )
                if pose_samples
                else []
            )
            next_delta = [next_pose[i] - current_pose[i] for i in range(min(len(current_pose), len(next_pose)))]

            rgb_histories: dict[str, list[Any]] = {}
            for key in config.alignment.rgb_source_keys:
                rgb_histories[key] = [sample.get(f"rgb_{key}_index", "") for sample in history_samples]

            phase_row = phase_by_sample[current_sample["sample_id"]]
            contact_row = contact_by_sample[current_sample["sample_id"]]
            window_rows.append(
                {
                    "window_id": f"{insert_id}::window::{target_index:05d}",
                    "insert_id": insert_id,
                    "trial_id": current_sample["trial_id"],
                    "split": split,
                    "target_sample_id": current_sample["sample_id"],
                    "target_sample_index": current_sample["sample_index"],
                    "history_sample_ids_json": json.dumps([sample["sample_id"] for sample in history_samples]),
                    "history_aligned_times_json": json.dumps([float(sample["aligned_time"]) for sample in history_samples]),
                    "ft_history_json": json.dumps(ft_history),
                    "pose_history_json": json.dumps(pose_history),
                    "structure_token": structure_token,
                    "y_phase": phase_row["y_phase"],
                    "y_contact": contact_row["y_contact"],
                    "y_success": insert_success,
                    "y_next_delta_json": json.dumps(next_delta),
                    "h5_path": h5_path,
                    "pose_path": insert_row["pose_path"],
                }
            )
            for key, values in rgb_histories.items():
                window_rows[-1][f"rgb_{key}_history_json"] = json.dumps(values)

    return window_rows


def write_window_rows(config: Any, window_rows: list[dict[str, Any]]) -> Path:
    """Persist window exports as one combined file plus split-specific subsets."""

    processed_root = ensure_directory(config.dataset.processed_root)
    output_path = processed_root / str(config.windows.window_index_name)
    fieldnames = [
        "window_id",
        "insert_id",
        "trial_id",
        "split",
        "target_sample_id",
        "target_sample_index",
        "history_sample_ids_json",
        "history_aligned_times_json",
        "ft_history_json",
        "pose_history_json",
        "structure_token",
        "y_phase",
        "y_contact",
        "y_success",
        "y_next_delta_json",
    ]
    fieldnames.extend(f"rgb_{key}_history_json" for key in config.alignment.rgb_source_keys)
    fieldnames.extend(["h5_path", "pose_path"])
    _write_csv(output_path, window_rows, fieldnames)

    for split_name in ("train", "val", "test"):
        split_rows = [row for row in window_rows if row["split"] == split_name]
        split_path = processed_root / str(getattr(config.windows.split_output_names, split_name))
        _write_csv(split_path, split_rows, fieldnames)

    return output_path


def run_window_export(config: Any) -> Path:
    """Run Milestone 6 window export end to end."""

    window_rows = build_window_rows(config)
    return write_window_rows(config, window_rows)
