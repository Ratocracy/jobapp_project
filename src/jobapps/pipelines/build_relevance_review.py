"""Build a blinded TF-IDF versus transformer relevance-review packet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pyspark.sql import functions as F

from jobapps.config import load_pipeline_config
from jobapps.ingestion import read_source
from jobapps.pipelines.jobs_pipeline import create_spark_session
from jobapps.pipelines.resumes_pipeline import run_resumes_pipeline
from jobapps.relevance_review import build_blinded_packet


REVIEW_FIELDS = [
    "review_id",
    "query_id",
    "resume_role",
    "resume_excerpt",
    "job_title",
    "company",
    "job_location",
    "job_level",
    "job_type",
    "job_summary_excerpt",
    "role_relevance_1_to_5",
    "skills_alignment_1_to_5",
    "seniority_alignment_1_to_5",
    "overall_relevance_1_to_5",
    "cannot_judge",
    "reviewer_notes",
]

ANSWER_FIELDS = [
    "review_id",
    "query_id",
    "resume_id",
    "job_link",
    "in_tfidf",
    "tfidf_rank",
    "tfidf_score",
    "in_transformer",
    "transformer_rank",
    "transformer_score",
]


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_instructions(path: Path, resume_count: int, row_count: int) -> None:
    path.write_text(
        f"""# Blinded relevance review

This packet contains {row_count} pooled candidate rows for {resume_count}
resume queries. Model identity, ranks, scores, labels, decision reasons, and
paired job descriptions are intentionally hidden.

Rate each candidate independently:

- `role_relevance_1_to_5`: 1 = unrelated role, 5 = strong role match.
- `skills_alignment_1_to_5`: 1 = little alignment, 5 = strong alignment.
- `seniority_alignment_1_to_5`: 1 = clear mismatch, 5 = strong alignment.
- `overall_relevance_1_to_5`: 1 = irrelevant, 5 = highly relevant.
- `cannot_judge`: enter `1` only when the supplied text is insufficient.
- `reviewer_notes`: optional short explanation.

Review all candidates for a query before moving to the next query. Do not open
`answer_key.csv` until ratings are complete. The answer key is needed only for
the final model comparison.
""",
        encoding="utf-8",
    )


def build_review_packet(
    config_path: str | Path,
    quality_path: str | Path,
    resume_count: int,
) -> tuple[Path, int]:
    config = load_pipeline_config(config_path)
    spark = create_spark_session(config)
    benchmark_dir = config.output.gold_dir / "benchmarks" / "transformer_1pct"
    output_dir = benchmark_dir / "relevance_review"
    try:
        tfidf = spark.read.parquet(str(benchmark_dir / "tfidf_top_k"))
        transformer = spark.read.parquet(str(benchmark_dir / "transformer_top_k"))
        available_ids = tfidf.select("resume_id").intersect(
            transformer.select("resume_id")
        )

        silver_resumes, _, _ = run_resumes_pipeline(
            spark,
            config,
            quality_path,
            write_output=False,
        )
        resume_rows = (
            silver_resumes.join(available_ids, on="resume_id", how="left_semi")
            .select(
                "resume_id",
                F.col("role_normalized").alias("resume_role"),
                F.substring("resume_normalized", 1, 1200).alias("resume_excerpt"),
            )
            .collect()
        )
        resume_metadata = {
            row["resume_id"]: {
                "resume_role": row["resume_role"] or "",
                "resume_excerpt": row["resume_excerpt"] or "",
            }
            for row in resume_rows
        }

        pooled_links = tfidf.select("job_link").union(
            transformer.select("job_link")
        ).distinct()
        postings = read_source(
            spark, config.input.raw_dir, "linkedin_job_postings"
        ).join(pooled_links, on="job_link", how="left_semi")
        summaries = read_source(
            spark, config.input.raw_dir, "job_summary"
        ).join(pooled_links, on="job_link", how="left_semi")
        job_rows = (
            postings.join(summaries, on="job_link", how="left")
            .select(
                "job_link",
                "job_title",
                "company",
                "job_location",
                "job_level",
                "job_type",
                F.substring("job_summary", 1, 1600).alias("job_summary_excerpt"),
            )
            .collect()
        )
        job_metadata = {
            row["job_link"]: {
                field: row[field] or ""
                for field in (
                    "job_title",
                    "company",
                    "job_location",
                    "job_level",
                    "job_type",
                    "job_summary_excerpt",
                )
            }
            for row in job_rows
        }

        reviewer_rows, answer_rows = build_blinded_packet(
            tfidf.collect(),
            transformer.collect(),
            resume_metadata,
            job_metadata,
            resume_count=resume_count,
            seed=config.runtime.random_seed,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "review_sheet.csv", reviewer_rows, REVIEW_FIELDS)
        _write_csv(output_dir / "answer_key.csv", answer_rows, ANSWER_FIELDS)
        _write_instructions(output_dir / "INSTRUCTIONS.md", resume_count, len(reviewer_rows))
        return output_dir, len(reviewer_rows)
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/transformer_1pct.yaml")
    parser.add_argument("--quality-config", default="config/data_quality.yaml")
    parser.add_argument("--resume-count", type=int, default=25)
    args = parser.parse_args()
    output_dir, row_count = build_review_packet(
        args.config,
        args.quality_config,
        args.resume_count,
    )
    print(f"Wrote {row_count} blinded review rows to {output_dir}")


if __name__ == "__main__":
    main()
