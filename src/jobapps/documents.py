"""Shared document contract for job and resume retrieval text."""

from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from jobapps.validation import require_columns


DOCUMENT_COLUMNS = ["document_id", "document_type", "source_id", "combined_text"]


def build_job_documents(silver_jobs: DataFrame) -> DataFrame:
    """Represent each usable job as title, summary, and normalized skills."""

    require_columns(
        silver_jobs,
        [
            "job_link",
            "job_title_normalized",
            "job_summary_normalized",
            "skills",
        ],
    )
    combined = F.concat_ws(
        " ",
        F.coalesce(F.col("job_title_normalized"), F.lit("")),
        F.coalesce(F.col("job_summary_normalized"), F.lit("")),
        F.coalesce(F.array_join(F.col("skills"), " "), F.lit("")),
    )
    return (
        silver_jobs
        .select(
            F.sha2(F.col("job_link"), 256).alias("document_id"),
            F.lit("job").alias("document_type"),
            F.col("job_link").alias("source_id"),
            F.trim(combined).alias("combined_text"),
        )
        .filter(F.length("combined_text") > 0)
    )


def build_resume_documents(silver_resumes: DataFrame) -> DataFrame:
    """Represent resumes without using their paired job description or label."""

    require_columns(silver_resumes, ["resume_id", "role_normalized", "resume_normalized"])
    combined = F.concat_ws(
        " ",
        F.coalesce(F.col("role_normalized"), F.lit("")),
        F.coalesce(F.col("resume_normalized"), F.lit("")),
    )
    return (
        silver_resumes
        .select(
            F.col("resume_id").alias("document_id"),
            F.lit("resume").alias("document_type"),
            F.col("resume_id").alias("source_id"),
            F.trim(combined).alias("combined_text"),
        )
        .filter(F.length("combined_text") > 0)
    )


def require_document_contract(documents: DataFrame) -> None:
    require_columns(documents, DOCUMENT_COLUMNS)
