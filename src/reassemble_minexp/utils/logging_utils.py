"""Lightweight experiment logging helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from reassemble_minexp.utils.paths import ensure_directory


def prepare_run_directory(output_root: str | Path, experiment_name: str) -> Path:
    """Create and return a stable run directory for an experiment."""

    return ensure_directory(Path(output_root) / experiment_name)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Persist a JSON payload with deterministic formatting."""

    destination = Path(path)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def write_resolved_config(path: str | Path, config: Any) -> Path:
    """Persist a fully resolved OmegaConf config."""

    destination = Path(path)
    destination.write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    return destination
