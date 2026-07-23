from pathlib import Path

import pytest

from jobapps.config import load_pipeline_config


def test_load_pipeline_config_resolves_project_paths() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "local.yaml"
    config = load_pipeline_config(path)

    assert config.input.raw_dir == config.project_root / "raw_data"
    assert config.output.silver_dir == config.project_root / "data" / "silver"
    assert config.runtime.sample_mode is True
    assert config.runtime.random_seed == 5110


def test_invalid_sample_fraction_is_rejected(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "bad.yaml"
    path.write_text(
        "input: {raw_dir: raw_data}\n"
        "output: {bronze_dir: b, silver_dir: s, gold_dir: g, quarantine_dir: q}\n"
        "runtime: {sample_mode: true, sample_fraction: 0, random_seed: 1}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample_fraction"):
        load_pipeline_config(path)
