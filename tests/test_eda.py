from jobapps.eda import document_length_distribution, job_type_counts


def test_job_type_counts_retains_unknown_values(spark) -> None:
    jobs = spark.createDataFrame(
        [("full-time",), ("full-time",), (None,)],
        ["job_type"],
    )

    rows = {row.job_type: row.job_count for row in job_type_counts(jobs).collect()}

    assert rows == {"full-time": 2, "unknown": 1}


def test_document_length_distribution_combines_document_types(spark) -> None:
    jobs = spark.createDataFrame(
        [("j1", "job", "j1", "word " * 120)],
        ["document_id", "document_type", "source_id", "combined_text"],
    )
    resumes = spark.createDataFrame(
        [("r1", "resume", "r1", "short text")],
        ["document_id", "document_type", "source_id", "combined_text"],
    )

    rows = {
        (row.document_type, row.length_bucket): row.document_count
        for row in document_length_distribution(jobs, resumes).collect()
    }

    assert rows[("job", "100-249")] == 1
    assert rows[("resume", "<100")] == 1
