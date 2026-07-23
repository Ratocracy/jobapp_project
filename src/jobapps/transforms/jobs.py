"""Silver-layer transformations for job-posting data."""

from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from jobapps.validation import require_columns


POSTING_COLUMNS = [
    "job_link",
    "last_processed_time",
    "got_summary",
    "got_ner",
    "is_being_worked",
    "job_title",
    "company",
    "job_location",
    "first_seen",
    "search_city",
    "search_country",
    "search_position",
    "job_level",
    "job_type",
]


def _normalized_text(column_name: str):
    return F.lower(F.regexp_replace(F.trim(F.col(column_name)), r"\s+", " "))


def transform_postings(postings: DataFrame) -> DataFrame:
    """Add typed and normalized columns while preserving raw source columns."""

    require_columns(postings, POSTING_COLUMNS)
    return (
        postings
        .withColumn("last_processed_at", F.to_timestamp("last_processed_time"))
        .withColumn("first_seen_date", F.to_date("first_seen"))
        .withColumn("got_summary_flag", F.col("got_summary") == F.lit("t"))
        .withColumn("got_ner_flag", F.col("got_ner") == F.lit("t"))
        .withColumn("is_being_worked_flag", F.col("is_being_worked") == F.lit("t"))
        .withColumn("job_title_normalized", _normalized_text("job_title"))
        .withColumn("company_normalized", _normalized_text("company"))
        .withColumn("job_location_normalized", _normalized_text("job_location"))
    )


def transform_summaries(summaries: DataFrame) -> DataFrame:
    """Normalize summaries without discarding their original text."""

    require_columns(summaries, ["job_link", "job_summary"])
    return summaries.withColumn(
        "job_summary_normalized",
        F.regexp_replace(F.trim(F.col("job_summary")), r"\s+", " "),
    )


def transform_skills(skills: DataFrame) -> DataFrame:
    """Parse comma-delimited skills into a normalized, unique array."""

    require_columns(skills, ["job_link", "job_skills"])
    parsed = F.expr(
        "filter(transform(split(coalesce(job_skills, ''), ','), "
        "skill -> lower(trim(skill))), skill -> skill <> '')"
    )
    return skills.withColumn("skills", F.array_sort(F.array_distinct(parsed)))


def build_silver_jobs(
    postings: DataFrame,
    summaries: DataFrame,
    skills: DataFrame,
    skill_count_warning: int = 100,
) -> DataFrame:
    """Create one enriched silver record per source job posting."""

    posting_rows = transform_postings(postings)
    summary_rows = transform_summaries(summaries)
    skill_rows = transform_skills(skills)

    return (
        posting_rows
        .join(summary_rows, on="job_link", how="left")
        .join(skill_rows, on="job_link", how="left")
        .withColumn("has_summary", F.col("job_summary").isNotNull())
        .withColumn(
            "has_skills",
            F.coalesce(F.size(F.col("skills")) > 0, F.lit(False)),
        )
        .withColumn("skill_count", F.coalesce(F.size(F.col("skills")), F.lit(0)))
        .withColumn("skills_count_warning", F.col("skill_count") > skill_count_warning)
        .withColumn(
            "eligible_for_ml",
            F.col("has_summary") & F.col("has_skills") & F.col("job_link").isNotNull(),
        )
    )
