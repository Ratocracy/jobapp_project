"""Configuration loading for local ETL runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InputConfig:
    raw_dir: Path


@dataclass(frozen=True)
class OutputConfig:
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    quarantine_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    sample_mode: bool
    sample_fraction: float
    random_seed: int
    spark_master: str
    driver_memory: str
    shuffle_partitions: int
    output_partitions: int
    resume_query_limit: int


@dataclass(frozen=True)
class NlpConfig:
    num_features: int
    min_document_frequency: int


@dataclass(frozen=True)
class RetrievalConfig:
    bucket_length: float
    num_hash_tables: int
    distance_threshold: float
    top_k: int


@dataclass(frozen=True)
class PipelineConfig:
    input: InputConfig
    output: OutputConfig
    runtime: RuntimeConfig
    nlp: NlpConfig
    retrieval: RetrievalConfig
    project_root: Path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return data


def _resolve(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load local pipeline settings and resolve paths from the project root."""

    config_path = Path(path).resolve()
    data = _read_yaml(config_path)
    project_root = config_path.parent.parent

    try:
        input_data = data["input"]
        output_data = data["output"]
        runtime_data = data["runtime"]
        nlp_data = data.get("nlp", {})
        retrieval_data = data.get("retrieval", {})
        sample_fraction = float(runtime_data["sample_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pipeline configuration: {config_path}") from exc

    if not 0 < sample_fraction <= 1:
        raise ValueError("runtime.sample_fraction must be greater than 0 and at most 1")
    output_partitions = int(runtime_data.get("output_partitions", 4))
    shuffle_partitions = int(runtime_data.get("shuffle_partitions", 16))
    if output_partitions < 1:
        raise ValueError("runtime.output_partitions must be at least 1")
    if shuffle_partitions < 1:
        raise ValueError("runtime.shuffle_partitions must be at least 1")
    resume_query_limit = int(runtime_data.get("resume_query_limit", 100))
    if resume_query_limit < 1:
        raise ValueError("runtime.resume_query_limit must be at least 1")

    num_features = int(nlp_data.get("num_features", 262_144))
    min_document_frequency = int(nlp_data.get("min_document_frequency", 2))
    bucket_length = float(retrieval_data.get("bucket_length", 0.8))
    num_hash_tables = int(retrieval_data.get("num_hash_tables", 3))
    distance_threshold = float(retrieval_data.get("distance_threshold", 1.2))
    top_k = int(retrieval_data.get("top_k", 10))
    if min(num_features, min_document_frequency, num_hash_tables, top_k) < 1:
        raise ValueError("NLP and retrieval integer settings must be at least 1")
    if bucket_length <= 0 or distance_threshold <= 0:
        raise ValueError("LSH bucket length and distance threshold must be positive")

    return PipelineConfig(
        input=InputConfig(raw_dir=_resolve(project_root, input_data["raw_dir"])),
        output=OutputConfig(
            bronze_dir=_resolve(project_root, output_data["bronze_dir"]),
            silver_dir=_resolve(project_root, output_data["silver_dir"]),
            gold_dir=_resolve(project_root, output_data["gold_dir"]),
            quarantine_dir=_resolve(project_root, output_data["quarantine_dir"]),
        ),
        runtime=RuntimeConfig(
            sample_mode=bool(runtime_data["sample_mode"]),
            sample_fraction=sample_fraction,
            random_seed=int(runtime_data["random_seed"]),
            spark_master=str(runtime_data.get("spark_master", "local[2]")),
            driver_memory=str(runtime_data.get("driver_memory", "4g")),
            shuffle_partitions=shuffle_partitions,
            output_partitions=output_partitions,
            resume_query_limit=resume_query_limit,
        ),
        nlp=NlpConfig(
            num_features=num_features,
            min_document_frequency=min_document_frequency,
        ),
        retrieval=RetrievalConfig(
            bucket_length=bucket_length,
            num_hash_tables=num_hash_tables,
            distance_threshold=distance_threshold,
            top_k=top_k,
        ),
        project_root=project_root,
    )


def load_quality_config(path: str | Path) -> dict[str, Any]:
    """Load the data-quality policy without interpreting individual rules."""

    return _read_yaml(Path(path).resolve())
