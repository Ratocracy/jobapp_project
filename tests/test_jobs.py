from jobapps.ingestion import deterministic_sample
from jobapps.schemas import JOB_SKILLS_SCHEMA, JOB_SUMMARY_SCHEMA, LINKEDIN_JOB_POSTINGS_SCHEMA
from jobapps.transforms.jobs import build_silver_jobs


def _posting(link: str, title: str = " Data  Scientist ") -> tuple[str, ...]:
    return (
        link,
        "2024-01-21 07:12:29+00:00",
        "t",
        "t",
        "f",
        title,
        " Example  Corp ",
        " Boston, MA ",
        "2024-01-14",
        "Boston",
        "United States",
        "Data Scientist",
        "Mid senior",
        "Hybrid",
    )


def test_build_silver_jobs_preserves_postings_and_adds_features(spark) -> None:
    postings = spark.createDataFrame(
        [_posting("job-1"), _posting("job-2", "Analyst")],
        LINKEDIN_JOB_POSTINGS_SCHEMA,
    )
    summaries = spark.createDataFrame(
        [("job-1", " Build   useful models. ")], JOB_SUMMARY_SCHEMA
    )
    skills = spark.createDataFrame(
        [("job-1", "Python, SQL, python,  ")], JOB_SKILLS_SCHEMA
    )

    result = build_silver_jobs(postings, summaries, skills)
    rows = {row.job_link: row for row in result.collect()}

    assert len(rows) == 2
    assert rows["job-1"].job_title == " Data  Scientist "
    assert rows["job-1"].job_title_normalized == "data scientist"
    assert rows["job-1"].skills == ["python", "sql"]
    assert rows["job-1"].skill_count == 2
    assert rows["job-1"].eligible_for_ml is True
    assert rows["job-2"].has_summary is False
    assert rows["job-2"].has_skills is False
    assert rows["job-2"].eligible_for_ml is False


def test_deterministic_sample_returns_same_keys(spark) -> None:
    frame = spark.createDataFrame([(f"job-{i}",) for i in range(100)], ["job_link"])

    first = deterministic_sample(frame, "job_link", 0.25, 5110)
    second = deterministic_sample(frame, "job_link", 0.25, 5110)

    assert sorted(row.job_link for row in first.collect()) == sorted(
        row.job_link for row in second.collect()
    )
