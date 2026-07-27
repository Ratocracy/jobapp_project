"""Spark aggregations used by final-project exploratory data analysis."""

from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from jobapps.documents import require_document_contract
from jobapps.validation import require_columns


LENGTH_BUCKETS = ["<100", "100-249", "250-499", "500-999", "1000+"]


def job_type_counts(silver_jobs: DataFrame) -> DataFrame:
    """Count sampled job postings by normalized job type in Spark."""

    require_columns(silver_jobs, ["job_type"])
    return (
        silver_jobs
        .select(
            F.coalesce(F.col("job_type"), F.lit("unknown")).alias("job_type")
        )
        .groupBy("job_type")
        .agg(F.count(F.lit(1)).alias("job_count"))
        .orderBy(F.desc("job_count"), F.asc("job_type"))
    )


def document_length_distribution(
    job_documents: DataFrame,
    resume_documents: DataFrame,
) -> DataFrame:
    """Aggregate job and resume word counts into interpretable length buckets."""

    for frame in (job_documents, resume_documents):
        require_document_contract(frame)
    documents = job_documents.unionByName(resume_documents)
    word_count = F.size(F.split(F.trim(F.col("combined_text")), r"\s+"))
    bucket = (
        F.when(word_count < 100, F.lit("<100"))
        .when(word_count < 250, F.lit("100-249"))
        .when(word_count < 500, F.lit("250-499"))
        .when(word_count < 1000, F.lit("500-999"))
        .otherwise(F.lit("1000+"))
    )
    return (
        documents
        .withColumn("length_bucket", bucket)
        .groupBy("document_type", "length_bucket")
        .agg(F.count(F.lit(1)).alias("document_count"))
    )


def job_quality_rates(silver_jobs: DataFrame) -> DataFrame:
    """Calculate summary and skills coverage as one Spark aggregate."""

    require_columns(silver_jobs, ["has_summary", "has_skills"])
    return silver_jobs.agg(
        F.count(F.lit(1)).alias("job_count"),
        F.avg(F.col("has_summary").cast("double")).alias("summary_coverage"),
        F.avg(F.col("has_skills").cast("double")).alias("skills_coverage"),
    )
