"""Pose JSON helpers for REASSEMBLE camera and board poses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_pose_metadata(json_path: str | Path) -> dict[str, Any]:
    """Read pose JSON and extract lightweight metadata.

    Official REASSEMBLE pose JSON files contain static camera and board poses,
    not time-series samples. Pose timing lives in the HDF5 timestamps group.
    """

    path = Path(json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return {
        "pose_path": str(path),
        "keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "start_time": None,
        "end_time": None,
        "num_pose_samples": 0,
        "num_pose_entities": len(payload) if isinstance(payload, dict) else 0,
    }
