"""Explicit schemas for raw source datasets.

These schemas intentionally mirror the raw Parquet files. Type normalization
(for example, converting ``t``/``f`` strings and timestamps) belongs in silver
transformations, not ingestion.
"""

from pyspark.sql.types import StringType, StructField, StructType


LINKEDIN_JOB_POSTINGS_SCHEMA = StructType(
    [
        StructField("job_link", StringType(), True),
        StructField("last_processed_time", StringType(), True),
        StructField("got_summary", StringType(), True),
        StructField("got_ner", StringType(), True),
        StructField("is_being_worked", StringType(), True),
        StructField("job_title", StringType(), True),
        StructField("company", StringType(), True),
        StructField("job_location", StringType(), True),
        StructField("first_seen", StringType(), True),
        StructField("search_city", StringType(), True),
        StructField("search_country", StringType(), True),
        StructField("search_position", StringType(), True),
        StructField("job_level", StringType(), True),
        StructField("job_type", StringType(), True),
    ]
)

JOB_SUMMARY_SCHEMA = StructType(
    [
        StructField("job_link", StringType(), True),
        StructField("job_summary", StringType(), True),
    ]
)

JOB_SKILLS_SCHEMA = StructType(
    [
        StructField("job_link", StringType(), True),
        StructField("job_skills", StringType(), True),
    ]
)

RESUME_SCREENING_SCHEMA = StructType(
    [
        StructField("Role", StringType(), True),
        StructField("Resume", StringType(), True),
        StructField("Decision", StringType(), True),
        StructField("Reason_for_decision", StringType(), True),
        StructField("Job_Description", StringType(), True),
    ]
)

SOURCE_SCHEMAS = {
    "linkedin_job_postings": LINKEDIN_JOB_POSTINGS_SCHEMA,
    "job_summary": JOB_SUMMARY_SCHEMA,
    "job_skills": JOB_SKILLS_SCHEMA,
    "resume_screening": RESUME_SCREENING_SCHEMA,
}


def get_source_schema(dataset_name: str) -> StructType:
    """Return the raw schema for a known dataset.

    Raises:
        KeyError: If ``dataset_name`` is not registered.
    """

    try:
        return SOURCE_SCHEMAS[dataset_name]
    except KeyError as exc:
        known = ", ".join(sorted(SOURCE_SCHEMAS))
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. Known datasets: {known}"
        ) from exc
