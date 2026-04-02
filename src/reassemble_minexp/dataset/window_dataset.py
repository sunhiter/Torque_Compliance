"""Window export for Milestone 6."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

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


def _prepare_window_context(config: Any) -> dict[str, Any]:
    processed_root = Path(config.dataset.processed_root)
    aligned_rows = _read_csv(processed_root / str(config.alignment.aligned_index_name))
    phase_rows = _read_csv(processed_root / str(config.labels.phase_index_name))
    contact_rows = _read_csv(processed_root / str(config.labels.contact_index_name))
    insert_rows = _read_csv(processed_root / str(config.dataset.insert_index_name))

    return {
        "aligned_by_insert": {
            insert_id: _sorted_samples(rows)
            for insert_id, rows in _group_rows(aligned_rows, "insert_id").items()
        },
        "phase_by_sample": {row["sample_id"]: row for row in phase_rows},
        "contact_by_sample": {row["sample_id"]: row for row in contact_rows},
        "insert_by_id": {row["insert_id"]: row for row in insert_rows},
    }


def _window_count(sample_count: int, history_length: int, history_step: int, export_stride: int) -> int:
    start_index = (history_length - 1) * history_step
    stop_index = sample_count - 1
    if sample_count < history_length or sample_count < 2 or start_index >= stop_index:
        return 0
    return ((stop_index - 1 - start_index) // export_stride) + 1


def _count_total_windows(aligned_by_insert: dict[str, list[dict[str, str]]], history_length: int, history_step: int, export_stride: int) -> int:
    return sum(
        _window_count(len(samples), history_length, history_step, export_stride)
        for samples in aligned_by_insert.values()
    )


def _precompute_sample_series(
    samples: list[dict[str, str]],
    pose_samples: list[list[float]],
    ft_samples: list[list[float]],
    rgb_source_keys: Iterable[str],
) -> dict[str, Any]:
    sample_ids = [sample["sample_id"] for sample in samples]
    aligned_times = [float(sample["aligned_time"]) for sample in samples]
    pose_vectors = [
        _interpolate_vector(
            pose_samples,
            int(sample["pose_index_0"]),
            int(sample["pose_index_1"]),
            float(sample["pose_weight_0"]),
            float(sample["pose_weight_1"]),
        )
        for sample in samples
    ] if pose_samples else []
    ft_vectors = [
        _interpolate_vector(
            ft_samples,
            int(sample["ft_index_0"]),
            int(sample["ft_index_1"]),
            float(sample["ft_weight_0"]),
            float(sample["ft_weight_1"]),
        )
        for sample in samples
    ] if ft_samples else []
    rgb_indices = {
        key: [sample.get(f"rgb_{key}_index", "") for sample in samples]
        for key in rgb_source_keys
    }
    return {
        "sample_ids": sample_ids,
        "aligned_times": aligned_times,
        "pose_vectors": pose_vectors,
        "ft_vectors": ft_vectors,
        "rgb_indices": rgb_indices,
    }


def _window_fieldnames(config: Any) -> list[str]:
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
    return fieldnames


def iter_window_rows(config: Any, context: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield fixed-length history windows from aligned samples and labels."""

    context = context or _prepare_window_context(config)
    aligned_by_insert = context["aligned_by_insert"]
    phase_by_sample = context["phase_by_sample"]
    contact_by_sample = context["contact_by_sample"]
    insert_by_id = context["insert_by_id"]

    history_length = int(config.windows.history_length)
    history_step = int(config.windows.history_step)
    export_stride = int(config.windows.export_stride)
    token_maps = _load_structure_tokens(getattr(config.windows, "structure_tokens_path", ""))
    split_lookup = _split_lookup(config)

    h5_cache: dict[str, dict[str, list[list[float]]]] = {}
    success_cache: dict[str, dict[str, Any]] = {}
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

        pose_source_key = samples[0]["pose_source_key"]
        ft_source_key = samples[0]["ft_source_key"]
        if pose_source_key or ft_source_key:
            with h5py.File(h5_path, "r") as handle:
                if pose_source_key and pose_source_key not in h5_cache[h5_path]:
                    h5_cache[h5_path][pose_source_key] = _read_numeric_dataset(handle, pose_source_key)
                if ft_source_key and ft_source_key not in h5_cache[h5_path]:
                    h5_cache[h5_path][ft_source_key] = _read_numeric_dataset(handle, ft_source_key)

        pose_samples = h5_cache[h5_path].get(pose_source_key, [])
        ft_samples = h5_cache[h5_path].get(ft_source_key, [])
        sample_series = _precompute_sample_series(samples, pose_samples, ft_samples, config.alignment.rgb_source_keys)

        for target_index in range((history_length - 1) * history_step, len(samples) - 1, export_stride):
            indices = _history_indices(target_index, history_length, history_step)
            next_sample = samples[target_index + 1]
            current_sample = samples[target_index]
            pose_history = [sample_series["pose_vectors"][index] for index in indices] if sample_series["pose_vectors"] else []
            ft_history = [sample_series["ft_vectors"][index] for index in indices] if sample_series["ft_vectors"] else []

            current_pose = sample_series["pose_vectors"][target_index] if sample_series["pose_vectors"] else []
            next_pose = sample_series["pose_vectors"][target_index + 1] if sample_series["pose_vectors"] else []
            next_delta = [next_pose[i] - current_pose[i] for i in range(min(len(current_pose), len(next_pose)))]

            phase_row = phase_by_sample[current_sample["sample_id"]]
            contact_row = contact_by_sample[current_sample["sample_id"]]
            row = {
                "window_id": f"{insert_id}::window::{target_index:05d}",
                "insert_id": insert_id,
                "trial_id": current_sample["trial_id"],
                "split": split,
                "target_sample_id": current_sample["sample_id"],
                "target_sample_index": current_sample["sample_index"],
                "history_sample_ids_json": json.dumps([sample_series["sample_ids"][index] for index in indices]),
                "history_aligned_times_json": json.dumps([sample_series["aligned_times"][index] for index in indices]),
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
            for key, values in sample_series["rgb_indices"].items():
                row[f"rgb_{key}_history_json"] = json.dumps([values[index] for index in indices])
            yield row


def build_window_rows(config: Any) -> list[dict[str, Any]]:
    """Build fixed-length history windows from aligned samples and labels."""

    return list(iter_window_rows(config))


def _format_eta(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "unknown"
    rounded = int(seconds + 0.5)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _print_progress(processed: int, total: int, start_time: float, *, final: bool = False) -> None:
    elapsed = max(time.monotonic() - start_time, 1e-9)
    rate = processed / elapsed
    if total > 0:
        percent = (processed / total) * 100.0
        remaining = max(total - processed, 0)
        eta_seconds = remaining / rate if rate > 0 else float("inf")
        message = (
            f"\rExporting windows: {processed}/{total} ({percent:5.1f}%)"
            f" | {rate:,.1f} windows/s | ETA {_format_eta(eta_seconds)}"
        )
    else:
        message = f"\rExporting windows: {processed} rows | {rate:,.1f} windows/s"

    end = "\n" if final else ""
    print(message, file=sys.stderr, end=end, flush=True)


def write_window_rows(config: Any, window_rows: Iterable[dict[str, Any]]) -> tuple[Path, int]:
    """Persist window exports as one combined file plus split-specific subsets."""

    processed_root = ensure_directory(config.dataset.processed_root)
    output_path = processed_root / str(config.windows.window_index_name)
    fieldnames = _window_fieldnames(config)

    split_paths = {
        split_name: processed_root / str(getattr(config.windows.split_output_names, split_name))
        for split_name in ("train", "val", "test")
    }
    row_count = 0

    with output_path.open("w", newline="", encoding="utf-8") as combined_handle:
        combined_writer = csv.DictWriter(combined_handle, fieldnames=fieldnames)
        combined_writer.writeheader()
        split_handles = {
            split_name: split_path.open("w", newline="", encoding="utf-8")
            for split_name, split_path in split_paths.items()
        }
        try:
            split_writers = {
                split_name: csv.DictWriter(handle, fieldnames=fieldnames)
                for split_name, handle in split_handles.items()
            }
            for writer in split_writers.values():
                writer.writeheader()

            for row in window_rows:
                combined_writer.writerow(row)
                split_name = row["split"]
                if split_name in split_writers:
                    split_writers[split_name].writerow(row)
                row_count += 1
        finally:
            for handle in split_handles.values():
                handle.close()

    return output_path, row_count


def export_windows(config: Any) -> tuple[Path, int]:
    """Run Milestone 6 window export and stream rows to disk."""

    context = _prepare_window_context(config)
    history_length = int(config.windows.history_length)
    history_step = int(config.windows.history_step)
    export_stride = int(config.windows.export_stride)
    total_windows = _count_total_windows(
        context["aligned_by_insert"],
        history_length,
        history_step,
        export_stride,
    )
    progress_every = max(1, int(getattr(config.windows, "progress_every", 500)))
    start_time = time.monotonic()

    def _progress_wrapped_rows() -> Iterator[dict[str, Any]]:
        processed = 0
        for row in iter_window_rows(config, context):
            processed += 1
            if processed == 1 or processed % progress_every == 0 or processed == total_windows:
                _print_progress(processed, total_windows, start_time)
            yield row
        _print_progress(processed, total_windows, start_time, final=True)

    return write_window_rows(config, _progress_wrapped_rows())


def run_window_export(config: Any) -> Path:
    """Run Milestone 6 window export end to end."""

    output_path, _ = export_windows(config)
    return output_path
