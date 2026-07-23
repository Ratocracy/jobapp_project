"""Executable MVP pipeline for the jobs silver table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

from pyspark.sql import DataFrame, SparkSession, functions as F

from jobapps.config import PipelineConfig, load_pipeline_config, load_quality_config
from jobapps.ingestion import deterministic_sample, read_source
from jobapps.transforms.jobs import build_silver_jobs
from jobapps.validation import count_duplicate_keys


@dataclass(frozen=True)
class PipelineMetrics:
    posting_count: int
    silver_count: int
    duplicate_job_links: int
    summary_coverage: float
    skills_coverage: float
    eligible_for_ml_count: int


def create_spark_session(config: PipelineConfig) -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder
        .master(config.runtime.spark_master)
        .appName("jobapps-silver-jobs")
        .config("spark.driver.memory", config.runtime.driver_memory)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", config.runtime.shuffle_partitions)
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .getOrCreate()
    )


def load_job_inputs(spark: SparkSession, config: PipelineConfig):
    postings = read_source(spark, config.input.raw_dir, "linkedin_job_postings")
    if config.runtime.sample_mode:
        postings = deterministic_sample(
            postings,
            key_column="job_link",
            fraction=config.runtime.sample_fraction,
            seed=config.runtime.random_seed,
        )

    selected_links = postings.select("job_link")
    summaries = (
        read_source(spark, config.input.raw_dir, "job_summary")
        .join(selected_links, on="job_link", how="left_semi")
    )
    skills = (
        read_source(spark, config.input.raw_dir, "job_skills")
        .join(selected_links, on="job_link", how="left_semi")
    )
    return postings, summaries, skills


def calculate_metrics(postings: DataFrame, silver: DataFrame) -> PipelineMetrics:
    posting_count = postings.count()
    aggregate = silver.agg(
        F.count(F.lit(1)).alias("silver_count"),
        F.sum(F.col("has_summary").cast("int")).alias("summary_count"),
        F.sum(F.col("has_skills").cast("int")).alias("skills_count"),
        F.sum(F.col("eligible_for_ml").cast("int")).alias("eligible_count"),
    ).first()
    silver_count = int(aggregate["silver_count"])
    summary_count = int(aggregate["summary_count"] or 0)
    skills_count = int(aggregate["skills_count"] or 0)
    eligible_count = int(aggregate["eligible_count"] or 0)
    return PipelineMetrics(
        posting_count=posting_count,
        silver_count=silver_count,
        duplicate_job_links=count_duplicate_keys(silver, ["job_link"]),
        summary_coverage=summary_count / silver_count if silver_count else 0.0,
        skills_coverage=skills_count / silver_count if silver_count else 0.0,
        eligible_for_ml_count=eligible_count,
    )


def run_jobs_pipeline(
    spark: SparkSession,
    config_path: str | Path,
    quality_path: str | Path,
    write_output: bool = True,
) -> tuple[DataFrame, PipelineMetrics, Path]:
    config = load_pipeline_config(config_path)
    quality = load_quality_config(quality_path)
    skill_warning = quality["datasets"]["job_skills"]["parsed_list_size"][
        "job_skills"
    ]["max_warning"]

    postings, summaries, skills = load_job_inputs(spark, config)
    postings = postings.cache()
    silver = build_silver_jobs(postings, summaries, skills, skill_warning).cache()
    metrics = calculate_metrics(postings, silver)

    if metrics.silver_count != metrics.posting_count:
        raise ValueError("Silver join changed the posting row count")
    if metrics.duplicate_job_links:
        raise ValueError("Silver output contains duplicate job_link values")

    output_name = "jobs_sample" if config.runtime.sample_mode else "jobs"
    output_path = config.output.silver_dir / output_name
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

    config = load_pipeline_config(args.config)
    spark = create_spark_session(config)
    try:
        silver, metrics, output_path = run_jobs_pipeline(
            spark,
            config_path=args.config,
            quality_path=args.quality_config,
            write_output=not args.no_write,
        )
        silver.printSchema()
        silver.select(
            "job_link",
            "job_title",
            "company",
            "job_type",
            "skill_count",
            "has_summary",
            "has_skills",
            "eligible_for_ml",
        ).show(10, truncate=60)
        print(metrics)
        if not args.no_write:
            print(f"Wrote silver jobs to: {output_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
