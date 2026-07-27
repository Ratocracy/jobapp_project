"""Build a blinded three-method relevance-review packet."""

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
    "in_ngram",
    "ngram_rank",
    "ngram_score",
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
        f"""# Three-method blinded relevance review

## Purpose

We are comparing three job-retrieval methods: unigram TF-IDF,
unigram-plus-bigram TF-IDF, and MiniLM transformer embeddings. This packet
contains {row_count} unique resume-job pairs for {resume_count} validation
resumes. Each pair appeared in the top 10 of at least one method.

The review is blinded. `review_sheet.csv` intentionally hides model identity,
rank, similarity score, resume ID, job link, prior hiring decisions, and
decision reasons. Do not attempt to identify which model produced a candidate.

## Files to distribute

Give reviewers:

- `review_sheet.csv`
- this `INSTRUCTIONS.md`

Do **not** distribute or open `answer_key.csv` until every rating is finalized.
It reveals model identities and would bias the review.

## Independent review by every group member

Every reviewer should independently label the complete sheet: all 100 query
IDs and all {row_count} candidate rows. Multiple independent ratings allow us
to measure reviewer agreement, average scores for each resume-job pair, and
investigate large disagreements.

1. Give each reviewer an identical untouched copy of `review_sheet.csv`.
2. Each person works independently and reviews Q001 through Q100.
3. Do not discuss individual candidates or compare scores until everyone has
   submitted a final file.
4. Save each completed copy as
   `review_sheet_<reviewer_name>.csv`.
5. Do not delete, reorder, or filter rows out of the submitted copy.

The project owner will join completed files by `review_id`, retain each
reviewer's ratings separately, calculate inter-rater agreement, and then
calculate consensus scores. For primary model evaluation, use the mean rating
across reviewers for each resume-job pair. Review or adjudicate pairs with
large rating differences instead of silently averaging obvious mistakes.

## Rating procedure

For each query:

1. Read `resume_role` and `resume_excerpt`.
2. Review all jobs for that query before finalizing scores.
3. Judge only information visible in the row.
4. Fill the four rating columns with whole numbers from 1 through 5.
5. Set `cannot_judge` to `1` only when the supplied text is insufficient;
   otherwise enter `0`.
6. Use `reviewer_notes` for short explanations, especially obvious mismatches,
   ambiguous cases, or `cannot_judge = 1`.

Do not edit identifiers, resume text, job text, or other source columns.

## Scoring rubric

### `role_relevance_1_to_5`

| Score | Meaning |
| ---: | --- |
| 1 | Unrelated occupation or function |
| 2 | Weakly related field but substantially different work |
| 3 | Plausible role match with notable differences |
| 4 | Strong role or functional match |
| 5 | Direct role match or highly appropriate next role |

### `skills_alignment_1_to_5`

| Score | Meaning |
| ---: | --- |
| 1 | Required skills are absent or incompatible |
| 2 | Only minor or generic skill overlap |
| 3 | Several relevant skills, with meaningful gaps |
| 4 | Most important skills align |
| 5 | Excellent alignment on core and specialized skills |

### `seniority_alignment_1_to_5`

| Score | Meaning |
| ---: | --- |
| 1 | Clearly incompatible experience or responsibility level |
| 2 | Large seniority mismatch |
| 3 | Uncertain or moderately mismatched |
| 4 | Appropriate seniority with minor uncertainty |
| 5 | Clear match in experience and responsibility level |

### `overall_relevance_1_to_5`

| Score | Meaning |
| ---: | --- |
| 1 | Irrelevant; should not be recommended |
| 2 | Weak recommendation |
| 3 | Plausible recommendation |
| 4 | Strong recommendation |
| 5 | Excellent recommendation |

Overall relevance is a holistic judgment, not necessarily the arithmetic
average of the other three scores. Location and employment type may inform the
score when clearly important, but do not invent preferences that are absent
from the resume excerpt.

## `cannot_judge`

Enter `1` only when missing or unusable text prevents a defensible rating.
Otherwise enter `0`. If `cannot_judge = 1`, leave the four rating columns blank
and briefly explain why in `reviewer_notes`.

Do not use `cannot_judge` merely because the decision is difficult.

## CSV handling

- Preserve the header names and column order.
- Preserve `review_id` and `query_id` exactly.
- Do not sort only part of the sheet.
- Do not add formulas, merged cells, or new columns.
- Save as CSV, not XLSX.
- If Excel warns that CSV features may be lost, choose to keep the CSV format.
- Do not paste resumes or job text into external AI systems.

## Completion check

Before returning a file, confirm:

- All 100 queries and all {row_count} rows are present.
- Every row is rated or marked `cannot_judge = 1`.
- Ratings contain only whole numbers 1-5.
- `cannot_judge` contains only 0 or 1.
- No identifier or source-text columns changed.
- The file opens successfully after saving.

Return the completed reviewer CSV to the project owner. The project owner will
validate and merge reviewer files before opening `answer_key.csv` and computing
model metrics.
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
    benchmark_dir = (
        config.output.gold_dir
        / "benchmarks"
        / f"transformer_1pct_{config.runtime.resume_query_split}"
    )
    ngram_dir = (
        config.output.gold_dir
        / "benchmarks"
        / f"ngram_1pct_{config.runtime.resume_query_split}"
    )
    output_dir = (
        config.output.gold_dir
        / "benchmarks"
        / f"three_method_1pct_{config.runtime.resume_query_split}"
        / "relevance_review"
    )
    try:
        tfidf = spark.read.parquet(str(ngram_dir / "unigram_top_k"))
        ngram = spark.read.parquet(str(ngram_dir / "ngram_top_k"))
        transformer = spark.read.parquet(str(benchmark_dir / "transformer_top_k"))
        available_ids = (
            tfidf.select("resume_id")
            .intersect(ngram.select("resume_id"))
            .intersect(transformer.select("resume_id"))
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

        pooled_links = (
            tfidf.select("job_link")
            .union(ngram.select("job_link"))
            .union(transformer.select("job_link"))
            .distinct()
        )
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
            ngram.collect(),
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
    parser.add_argument("--resume-count", type=int, default=100)
    args = parser.parse_args()
    output_dir, row_count = build_review_packet(
        args.config,
        args.quality_config,
        args.resume_count,
    )
    print(f"Wrote {row_count} blinded review rows to {output_dir}")


if __name__ == "__main__":
    main()
