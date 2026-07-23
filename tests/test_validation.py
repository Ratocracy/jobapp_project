import pytest

from jobapps.validation import count_duplicate_keys, require_columns


def test_require_columns_reports_missing_columns(spark) -> None:
    frame = spark.createDataFrame([("job-1",)], ["job_link"])

    with pytest.raises(ValueError, match="job_title"):
        require_columns(frame, ["job_link", "job_title"])


def test_count_duplicate_keys_counts_extra_rows(spark) -> None:
    frame = spark.createDataFrame(
        [("job-1",), ("job-1",), ("job-1",), ("job-2",)], ["job_link"]
    )

    assert count_duplicate_keys(frame, ["job_link"]) == 2
