"""Tests for raw source schema declarations."""

import pytest
from pyspark.sql.types import StringType, StructType

from jobapps.schemas import SOURCE_SCHEMAS, get_source_schema


EXPECTED_COLUMNS = {
    "linkedin_job_postings": [
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
    ],
    "job_summary": ["job_link", "job_summary"],
    "job_skills": ["job_link", "job_skills"],
    "resume_screening": [
        "Role",
        "Resume",
        "Decision",
        "Reason_for_decision",
        "Job_Description",
    ],
}


@pytest.mark.parametrize("dataset_name", EXPECTED_COLUMNS)
def test_source_schema_columns_match_raw_contract(dataset_name: str) -> None:
    schema = get_source_schema(dataset_name)

    assert isinstance(schema, StructType)
    assert schema.fieldNames() == EXPECTED_COLUMNS[dataset_name]
    assert all(isinstance(field.dataType, StringType) for field in schema.fields)


def test_all_expected_datasets_are_registered() -> None:
    assert set(SOURCE_SCHEMAS) == set(EXPECTED_COLUMNS)


def test_unknown_dataset_raises_helpful_error() -> None:
    with pytest.raises(KeyError, match="Unknown dataset 'missing'"):
        get_source_schema("missing")
