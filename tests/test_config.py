from pathlib import Path

from reassemble_minexp.utils.config import load_config


def test_load_config_supports_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text("dataset:\n  raw_root: data/raw\n", encoding="utf-8")

    config = load_config(config_path, ["dataset.raw_root=data/custom"])

    assert config.dataset.raw_root == "data/custom"
