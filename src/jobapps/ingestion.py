"""Schema-driven readers and deterministic development sampling."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, functions as F

from jobapps.schemas import get_source_schema


SOURCE_FILES = {
    "linkedin_job_postings": "linkedin_job_postings.parquet",
    "job_summary": "job_summary.parquet",
    "job_skills": "job_skills.parquet",
    "resume_screening": "resume_screening_train.parquet",
}


def read_source(spark: SparkSession, raw_dir: Path, dataset_name: str) -> DataFrame:
    """Read a registered Parquet source with its explicit raw schema."""

    try:
        filename = SOURCE_FILES[dataset_name]
    except KeyError as exc:
        known = ", ".join(sorted(SOURCE_FILES))
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. Known datasets: {known}"
        ) from exc

    path = raw_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Source dataset not found: {path}")
    return spark.read.schema(get_source_schema(dataset_name)).parquet(str(path))


def deterministic_sample(
    dataframe: DataFrame,
    key_column: str,
    fraction: float,
    seed: int,
    buckets: int = 1_000_000,
) -> DataFrame:
    """Select a stable approximate fraction based on a key hash."""

    if key_column not in dataframe.columns:
        raise ValueError(f"Sampling key is missing: {key_column}")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")
    if fraction == 1:
        return dataframe

    threshold = max(1, round(fraction * buckets))
    bucket = F.pmod(F.xxhash64(F.col(key_column), F.lit(seed)), F.lit(buckets))
    return dataframe.filter(bucket < F.lit(threshold))
