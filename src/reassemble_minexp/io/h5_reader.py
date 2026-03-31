"""HDF5 metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py


def _convert_value(value: Any) -> Any:
    """Convert HDF5 scalar values into plain Python types."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value.tolist()
    return value


def _group_to_python(group: h5py.Group) -> dict[str, Any]:
    """Convert an HDF5 group into a nested dictionary."""

    payload: dict[str, Any] = {"attrs": {key: _convert_value(val) for key, val in group.attrs.items()}}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            payload[key] = _group_to_python(value)
        else:
            payload[key] = _convert_value(value[()])
    return payload


def _collect_available_paths(group: h5py.Group, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else key
        paths.append(path)
        if isinstance(value, h5py.Group):
            paths.extend(_collect_available_paths(value, path))
    return paths


def _dataset_edge_values(dataset: h5py.Dataset) -> tuple[Any, Any] | None:
    if dataset.shape == ():
        value = _convert_value(dataset[()])
        return value, value
    if not dataset.shape or dataset.shape[0] == 0:
        return None
    first = _convert_value(dataset[0])
    last = _convert_value(dataset[-1])
    if isinstance(first, list):
        first = first[0] if first else None
    if isinstance(last, list):
        last = last[-1] if last else None
    return first, last


def _collect_timestamp_ranges(group: h5py.Group, prefix: str = "") -> dict[str, dict[str, Any]]:
    ranges: dict[str, dict[str, Any]] = {}
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Group):
            ranges.update(_collect_timestamp_ranges(value, path))
            continue
        edge_values = _dataset_edge_values(value)
        if edge_values is None:
            continue
        start_time, end_time = edge_values
        ranges[path] = {"start_time": start_time, "end_time": end_time}
    return ranges


def read_h5_metadata(h5_path: str | Path, segments_key: str) -> dict[str, Any]:
    """Read lightweight trial metadata from one HDF5 file."""

    path = Path(h5_path)
    metadata: dict[str, Any] = {
        "h5_path": str(path),
        "segments_info": None,
        "attrs": {},
        "top_level_keys": [],
        "available_paths": [],
        "timestamp_ranges": {},
    }

    with h5py.File(path, "r") as handle:
        metadata["attrs"] = {key: _convert_value(value) for key, value in handle.attrs.items()}
        metadata["top_level_keys"] = list(handle.keys())
        metadata["available_paths"] = _collect_available_paths(handle)

        if "timestamps" in handle and isinstance(handle["timestamps"], h5py.Group):
            metadata["timestamp_ranges"] = _collect_timestamp_ranges(handle["timestamps"])

        if segments_key in handle:
            node = handle[segments_key]
            if isinstance(node, h5py.Dataset):
                raw = node[()]
                if isinstance(raw, bytes):
                    metadata["segments_info"] = json.loads(raw.decode("utf-8"))
                elif isinstance(raw, str):
                    metadata["segments_info"] = json.loads(raw)
                else:
                    metadata["segments_info"] = raw.tolist()
            else:
                metadata["segments_info"] = _group_to_python(node)

    return metadata
