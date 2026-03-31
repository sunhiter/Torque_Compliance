"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: Iterable[str] | None = None) -> DictConfig:
    """Load a YAML config and apply optional OmegaConf dotlist overrides."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = OmegaConf.load(path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        config = OmegaConf.merge(config, override_cfg)
    return config
