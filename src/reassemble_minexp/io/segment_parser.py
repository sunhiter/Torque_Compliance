"""Segment metadata parsing helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


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


def _normalize_segment_collection(segments_info: Any) -> list[dict[str, Any]]:
    if isinstance(segments_info, dict):
        if "segments" in segments_info and isinstance(segments_info["segments"], list):
            return [item for item in segments_info["segments"] if isinstance(item, dict)]
        if "high_level" in segments_info and isinstance(segments_info["high_level"], list):
            return [item for item in segments_info["high_level"] if isinstance(item, dict)]
        keyed_segments = [
            value
            for key, value in segments_info.items()
            if key != "attrs" and isinstance(value, dict)
        ]
        if keyed_segments and len(keyed_segments) == len([key for key in segments_info if key != "attrs"]):
            return keyed_segments
        return [segments_info]
    if isinstance(segments_info, list):
        return [item for item in segments_info if isinstance(item, dict)]
    return []


def _low_level_segments(segment: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("low_level", "Low_level", "low_level_segments", "children", "subsegments"):
        items = segment.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        if isinstance(items, dict):
            return [value for name, value in items.items() if name != "attrs" and isinstance(value, dict)]
    return []


def _segment_name(segment: dict[str, Any], fallback: str) -> str:
    return str(_segment_value(segment, ["name", "label", "skill", "action", "text"]) or fallback)


def parse_segment_rows(segments_info: Any, trial_id: str, schema_cfg: Any) -> list[dict[str, Any]]:
    """Flatten high-level and low-level segment metadata into CSV-friendly rows."""

    if not segments_info:
        return []

    high_level_items = _normalize_segment_collection(segments_info)
    rows: list[dict[str, Any]] = []
    timestamp_cfg = schema_cfg.timestamp_keys

    for high_index, high_segment in enumerate(high_level_items):
        high_name = _segment_name(high_segment, fallback=f"segment_{high_index}")
        high_id = str(_segment_value(high_segment, ["id"]) or high_index)
        rows.append(
            {
                "trial_id": trial_id,
                "segment_level": "high",
                "segment_id": high_id,
                "parent_segment_id": "",
                "segment_name": high_name,
                "start_time": _segment_value(high_segment, list(timestamp_cfg.start)),
                "end_time": _segment_value(high_segment, list(timestamp_cfg.end)),
            }
        )

        for low_index, low_segment in enumerate(_low_level_segments(high_segment)):
            low_id = _segment_value(low_segment, ["id"]) or low_index
            rows.append(
                {
                    "trial_id": trial_id,
                    "segment_level": "low",
                    "segment_id": f"{high_id}::{low_id}",
                    "parent_segment_id": high_id,
                    "segment_name": _segment_name(low_segment, fallback=f"{high_name}_low_{low_index}"),
                    "start_time": _segment_value(low_segment, list(timestamp_cfg.start)),
                    "end_time": _segment_value(low_segment, list(timestamp_cfg.end)),
                }
            )

    return rows


def summarize_trial_segments(segment_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute lightweight per-trial segment summaries."""

    rows = list(segment_rows)
    high_names = [row["segment_name"] for row in rows if row["segment_level"] == "high"]
    low_names = [row["segment_name"] for row in rows if row["segment_level"] == "low"]
    start_times = [row["start_time"] for row in rows if row["start_time"] is not None]
    end_times = [row["end_time"] for row in rows if row["end_time"] is not None]

    return {
        "num_high_level_segments": len(high_names),
        "num_low_level_segments": len(low_names),
        "high_level_sequence": "|".join(high_names),
        "low_level_sequence": "|".join(low_names),
        "segment_start_time": min(start_times) if start_times else None,
        "segment_end_time": max(end_times) if end_times else None,
    }
