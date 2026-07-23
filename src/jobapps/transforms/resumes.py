"""Silver-layer transformations for resume-screening data."""

from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from jobapps.validation import require_columns


RESUME_COLUMNS = [
    "Role",
    "Resume",
    "Decision",
    "Reason_for_decision",
    "Job_Description",
]


def normalize_text(column):
    """Lowercase and collapse whitespace in a Spark string column."""

    return F.lower(F.regexp_replace(F.trim(column), r"\s+", " "))


def mask_direct_identifiers(column):
    """Mask common direct identifiers using deterministic Spark regex rules.

    This is an MVP safeguard, not a replacement for a production PII detector.
    """

    masked = F.regexp_replace(
        column,
        r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        "[EMAIL]",
    )
    masked = F.regexp_replace(
        masked,
        r"(?i)(?:https?://)?(?:www\.)?linkedin\.com/\S+",
        "[LINKEDIN]",
    )
    masked = F.regexp_replace(
        masked,
        r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)",
        "[PHONE]",
    )
    return F.regexp_replace(
        masked,
        r"(?i)(professional resume for)\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,3}):(?:\s*\2)?",
        "$1 [NAME]: [NAME]",
    )


def assign_group_split(
    dataframe: DataFrame,
    group_column: str,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> DataFrame:
    """Assign all records in the same group to a deterministic data split."""

    bucket = F.pmod(
        F.xxhash64(F.col(group_column), F.lit(seed)), F.lit(1_000_000)
    ) / F.lit(1_000_000.0)
    return dataframe.withColumn(
        "split",
        F.when(bucket < train_ratio, F.lit("train"))
        .when(bucket < train_ratio + validation_ratio, F.lit("validation"))
        .otherwise(F.lit("test")),
    )


def build_silver_resumes(
    resumes: DataFrame,
    seed: int = 5110,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> DataFrame:
    """Create stable IDs, PII-masked text, labels, and leakage-safe splits."""

    require_columns(resumes, RESUME_COLUMNS)
    result = (
        resumes
        .withColumn("role_normalized", normalize_text(F.col("Role")))
        .withColumn(
            "job_description_normalized",
            normalize_text(F.col("Job_Description")),
        )
        .withColumn("resume_masked", mask_direct_identifiers(F.col("Resume")))
        .withColumn("resume_normalized", normalize_text(F.col("resume_masked")))
        .withColumn("decision_normalized", normalize_text(F.col("Decision")))
        .withColumn("label", (F.col("decision_normalized") == "select").cast("int"))
        .withColumn("resume_id", F.sha2(F.col("Resume"), 256))
        .withColumn(
            "job_description_id",
            F.sha2(F.col("job_description_normalized"), 256),
        )
        .withColumn(
            "pii_masking_warning",
            F.col("resume_normalized").rlike(
                r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"
            ),
        )
    )
    return assign_group_split(
        result,
        group_column="job_description_id",
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
