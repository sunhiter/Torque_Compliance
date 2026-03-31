"""Dataset scanning and trial-index construction."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reassemble_minexp.io.h5_reader import read_h5_metadata
from reassemble_minexp.io.pose_json_reader import read_pose_metadata
from reassemble_minexp.io.segment_parser import parse_segment_rows, summarize_trial_segments
from reassemble_minexp.utils.paths import ensure_directory


@dataclass(slots=True)
class ScanArtifacts:
    """CSV output paths created by the scan and index pipeline."""

    manifest_path: Path
    trial_index_path: Path
    segment_index_path: Path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _first_present(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _has_any_key(h5_metadata: dict[str, Any], attrs: dict[str, Any], candidate_keys: list[str]) -> bool:
    top_level_keys = set(h5_metadata.get("top_level_keys", []))
    available_paths = set(h5_metadata.get("available_paths", []))
    attr_keys = set(attrs.keys())
    leaf_keys = {path.rsplit("/", maxsplit=1)[-1] for path in available_paths}
    return any(key in top_level_keys or key in attr_keys or key in available_paths or key in leaf_keys for key in candidate_keys)


def _timestamp_range(h5_metadata: dict[str, Any], candidate_keys: list[str]) -> tuple[Any, Any]:
    timestamp_ranges = h5_metadata.get("timestamp_ranges", {})
    for key in candidate_keys:
        if key in timestamp_ranges:
            window = timestamp_ranges[key]
            return window.get("start_time"), window.get("end_time")
    for path, window in timestamp_ranges.items():
        if path.rsplit("/", maxsplit=1)[-1] in candidate_keys:
            return window.get("start_time"), window.get("end_time")
    return None, None


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


def _segment_text(segment: dict[str, Any]) -> str | None:
    value = _segment_value(segment, ["text", "name", "label", "skill", "action"])
    if value in (None, ""):
        return None
    return str(value).strip()


def _infer_object_name(segments_info: Any) -> str | None:
    objects: list[str] = []
    for segment in _high_level_segments(segments_info):
        text = _segment_text(segment)
        if not text or text.lower().startswith("no action"):
            continue
        match = re.match(r"^(pick|insert|remove|place)\s+(.+?)[.!]?$", text, flags=re.IGNORECASE)
        if not match:
            continue
        object_name = match.group(2).strip()
        if object_name and object_name not in objects:
            objects.append(object_name)
    return "|".join(objects) if objects else None


def _infer_trial_success(segments_info: Any, result_keys: list[str]) -> Any:
    meaningful_successes: list[bool] = []
    fallback_successes: list[bool] = []

    for segment in _high_level_segments(segments_info):
        value = _normalize_bool(_segment_value(segment, result_keys))
        if not isinstance(value, bool):
            continue
        fallback_successes.append(value)
        text = _segment_text(segment)
        if text and not text.lower().startswith("no action"):
            meaningful_successes.append(value)

    if meaningful_successes:
        return all(meaningful_successes)
    if fallback_successes:
        return all(fallback_successes)
    return None


def scan_raw_files(config: Any) -> list[dict[str, Any]]:
    """Create manifest rows for HDF5 trials and matching pose JSON files."""

    raw_root = Path(config.dataset.raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_root}")

    manifest_rows: list[dict[str, Any]] = []
    pose_suffix = str(config.dataset.pose_suffix)
    pose_root = Path(getattr(config.dataset, "pose_root", raw_root))

    for h5_path in sorted(raw_root.rglob(str(config.dataset.h5_glob))):
        trial_id = h5_path.stem
        relative_parent = h5_path.relative_to(raw_root).parent
        pose_path = pose_root / relative_parent / f"{trial_id}{pose_suffix}"
        manifest_rows.append(
            {
                "trial_id": trial_id,
                "h5_path": str(h5_path),
                "pose_path": str(pose_path) if pose_path.exists() else "",
                "has_h5": True,
                "has_pose_json": pose_path.exists(),
            }
        )

    return manifest_rows


def write_manifest(config: Any, manifest_rows: list[dict[str, Any]]) -> Path:
    """Persist the file manifest CSV."""

    processed_root = ensure_directory(config.dataset.processed_root)
    manifest_path = processed_root / str(config.dataset.manifest_name)
    _write_csv(
        manifest_path,
        manifest_rows,
        ["trial_id", "h5_path", "pose_path", "has_h5", "has_pose_json"],
    )
    return manifest_path


def build_trial_index(config: Any, manifest_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build trial-level rows and flattened segment rows from the manifest."""

    trial_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    for manifest_row in manifest_rows:
        h5_metadata = read_h5_metadata(manifest_row["h5_path"], config.schema.segments_key)
        segments_info = h5_metadata.get("segments_info")
        pose_metadata = read_pose_metadata(manifest_row["pose_path"]) if manifest_row["has_pose_json"] else {}
        trial_segment_rows = parse_segment_rows(segments_info, manifest_row["trial_id"], config.schema)
        segment_rows.extend(trial_segment_rows)
        segment_summary = summarize_trial_segments(trial_segment_rows)
        attrs = h5_metadata.get("attrs", {})
        pose_start_time, pose_end_time = _timestamp_range(h5_metadata, list(config.schema.modality_keys.pose))
        object_name = _first_present(attrs, list(config.schema.object_keys)) or _infer_object_name(segments_info)
        success = _normalize_bool(_first_present(attrs, list(config.schema.result_keys)))
        if success is None:
            success = _infer_trial_success(segments_info, list(config.schema.result_keys))

        trial_rows.append(
            {
                "trial_id": manifest_row["trial_id"],
                "h5_path": manifest_row["h5_path"],
                "pose_path": pose_metadata.get("pose_path", manifest_row["pose_path"]),
                "object_name": object_name,
                "success": success,
                "has_ft": _has_any_key(h5_metadata, attrs, list(config.schema.modality_keys.ft)),
                "has_pose": _has_any_key(h5_metadata, attrs, list(config.schema.modality_keys.pose)),
                "has_rgb": _has_any_key(h5_metadata, attrs, list(config.schema.modality_keys.rgb)),
                "pose_start_time": pose_start_time,
                "pose_end_time": pose_end_time,
                "segment_start_time": segment_summary["segment_start_time"],
                "segment_end_time": segment_summary["segment_end_time"],
                "num_high_level_segments": segment_summary["num_high_level_segments"],
                "num_low_level_segments": segment_summary["num_low_level_segments"],
                "high_level_sequence": segment_summary["high_level_sequence"],
                "low_level_sequence": segment_summary["low_level_sequence"],
            }
        )

    return trial_rows, segment_rows


def write_trial_outputs(config: Any, trial_rows: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> ScanArtifacts:
    """Persist trial and segment CSV outputs."""

    processed_root = ensure_directory(config.dataset.processed_root)
    trial_index_path = processed_root / str(config.dataset.trial_index_name)
    segment_index_path = processed_root / str(config.dataset.segment_index_name)

    _write_csv(
        trial_index_path,
        trial_rows,
        [
            "trial_id",
            "h5_path",
            "pose_path",
            "object_name",
            "success",
            "has_ft",
            "has_pose",
            "has_rgb",
            "pose_start_time",
            "pose_end_time",
            "segment_start_time",
            "segment_end_time",
            "num_high_level_segments",
            "num_low_level_segments",
            "high_level_sequence",
            "low_level_sequence",
        ],
    )
    _write_csv(
        segment_index_path,
        segment_rows,
        [
            "trial_id",
            "segment_level",
            "segment_id",
            "parent_segment_id",
            "segment_name",
            "start_time",
            "end_time",
        ],
    )
    return ScanArtifacts(
        manifest_path=processed_root / str(config.dataset.manifest_name),
        trial_index_path=trial_index_path,
        segment_index_path=segment_index_path,
    )


def run_scan_pipeline(config: Any) -> ScanArtifacts:
    """Run Milestone 2 scanning and indexing end to end."""

    manifest_rows = scan_raw_files(config)
    manifest_path = write_manifest(config, manifest_rows)
    trial_rows, segment_rows = build_trial_index(config, manifest_rows)
    artifacts = write_trial_outputs(config, trial_rows, segment_rows)
    return ScanArtifacts(
        manifest_path=manifest_path,
        trial_index_path=artifacts.trial_index_path,
        segment_index_path=artifacts.segment_index_path,
    )
