"""Generate required EDA figures from Spark-aggregated sampled data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jobapps.config import load_pipeline_config
from jobapps.documents import build_job_documents, build_resume_documents
from jobapps.eda import (
    LENGTH_BUCKETS,
    document_length_distribution,
    job_quality_rates,
    job_type_counts,
)
from jobapps.pipelines.jobs_pipeline import create_spark_session, run_jobs_pipeline
from jobapps.pipelines.resumes_pipeline import run_resumes_pipeline


def _plot_job_types(rows, output_path: Path) -> None:
    top = rows[:12]
    labels = [row["job_type"] for row in reversed(top)]
    counts = [row["job_count"] for row in reversed(top)]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(labels, counts, color="#355c7d")
    axis.set_title("Top job types in deterministic 1% catalog sample")
    axis.set_xlabel("Job postings (log scale)")
    axis.set_xscale("log")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_document_lengths(rows, output_path: Path) -> None:
    counts = {
        (row["document_type"], row["length_bucket"]): row["document_count"]
        for row in rows
    }
    positions = np.arange(len(LENGTH_BUCKETS))
    width = 0.38
    job_values = [counts.get(("job", bucket), 0) for bucket in LENGTH_BUCKETS]
    resume_values = [counts.get(("resume", bucket), 0) for bucket in LENGTH_BUCKETS]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(positions - width / 2, job_values, width, label="Jobs", color="#355c7d")
    axis.bar(
        positions + width / 2,
        resume_values,
        width,
        label="Resumes",
        color="#c06c84",
    )
    axis.set_yscale("log")
    axis.set_xticks(positions, LENGTH_BUCKETS)
    axis.set_xlabel("Combined-text word count")
    axis.set_ylabel("Documents (log scale)")
    axis.set_title("Job and resume document-length distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def generate_eda(
    config_path: str | Path,
    quality_path: str | Path,
    output_dir: str | Path,
) -> dict:
    config = load_pipeline_config(config_path)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    spark = create_spark_session(config)
    try:
        silver_jobs, _, _ = run_jobs_pipeline(
            spark, config_path, quality_path, write_output=False
        )
        silver_resumes, _, _ = run_resumes_pipeline(
            spark, config, quality_path, write_output=False
        )
        job_documents = build_job_documents(silver_jobs)
        resume_documents = build_resume_documents(silver_resumes)

        type_rows = job_type_counts(silver_jobs).collect()
        length_rows = document_length_distribution(
            job_documents, resume_documents
        ).collect()
        quality = job_quality_rates(silver_jobs).first().asDict()

        _plot_job_types(type_rows, output_path / "job_type_counts.png")
        _plot_document_lengths(
            length_rows,
            output_path / "document_length_distribution.png",
        )
        summary = {
            "sample_fraction": config.runtime.sample_fraction,
            "job_count": int(quality["job_count"]),
            "summary_coverage": round(float(quality["summary_coverage"]), 6),
            "skills_coverage": round(float(quality["skills_coverage"]), 6),
            "resume_count": silver_resumes.count(),
        }
        (output_path / "eda_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/ngram_1pct.yaml")
    parser.add_argument("--quality-config", default="config/data_quality.yaml")
    parser.add_argument("--output-dir", default="reports/eda")
    args = parser.parse_args()
    print(generate_eda(args.config, args.quality_config, args.output_dir))


if __name__ == "__main__":
    main()
