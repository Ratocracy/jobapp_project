"""Executable pipeline for the resume silver table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, functions as F

from jobapps.config import PipelineConfig, load_pipeline_config, load_quality_config
from jobapps.ingestion import read_source
from jobapps.transforms.resumes import build_silver_resumes
from jobapps.validation import count_duplicate_keys


@dataclass(frozen=True)
class ResumeMetrics:
    source_count: int
    silver_count: int
    duplicate_resume_ids: int
    selected_count: int
    pii_warning_count: int


def calculate_resume_metrics(source: DataFrame, silver: DataFrame) -> ResumeMetrics:
    source_count = source.count()
    aggregate = silver.agg(
        F.count(F.lit(1)).alias("silver_count"),
        F.sum(F.col("label")).alias("selected_count"),
        F.sum(F.col("pii_masking_warning").cast("int")).alias("pii_warning_count"),
    ).first()
    return ResumeMetrics(
        source_count=source_count,
        silver_count=int(aggregate["silver_count"]),
        duplicate_resume_ids=count_duplicate_keys(silver, ["resume_id"]),
        selected_count=int(aggregate["selected_count"] or 0),
        pii_warning_count=int(aggregate["pii_warning_count"] or 0),
    )


def run_resumes_pipeline(
    spark: SparkSession,
    config: PipelineConfig,
    quality_path: str | Path,
    write_output: bool = True,
) -> tuple[DataFrame, ResumeMetrics, Path]:
    quality = load_quality_config(quality_path)
    split = quality["resume_splits"]
    source = read_source(spark, config.input.raw_dir, "resume_screening").cache()
    silver = build_silver_resumes(
        source,
        seed=int(split["random_seed"]),
        train_ratio=float(split["ratios"]["train"]),
        validation_ratio=float(split["ratios"]["validation"]),
    ).cache()
    metrics = calculate_resume_metrics(source, silver)
    if metrics.source_count != metrics.silver_count:
        raise ValueError("Resume silver transformation changed the row count")
    if metrics.duplicate_resume_ids:
        raise ValueError("Resume silver output contains duplicate resume_id values")

    output_path = config.output.silver_dir / "resumes"
    if write_output:
        (
            silver.coalesce(config.runtime.output_partitions)
            .write.mode("overwrite")
            .parquet(str(output_path))
        )
    return silver, metrics, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--quality-config", default="config/data_quality.yaml")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    from jobapps.pipelines.jobs_pipeline import create_spark_session

    config = load_pipeline_config(args.config)
    spark = create_spark_session(config)
    try:
        silver, metrics, output_path = run_resumes_pipeline(
            spark, config, args.quality_config, write_output=not args.no_write
        )
        silver.select(
            "resume_id", "Role", "Decision", "label", "split", "pii_masking_warning"
        ).show(10, truncate=40)
        print(metrics)
        if not args.no_write:
            print(f"Wrote silver resumes to: {output_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
