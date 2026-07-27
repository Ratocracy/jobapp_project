from jobapps.documents import (
    build_job_documents,
    build_resume_documents,
    select_resume_queries,
)


def test_shared_documents_use_allowed_fields_only(spark) -> None:
    jobs = spark.createDataFrame(
        [("job-1", "data scientist", "build models", ["python", "spark"])],
        ["job_link", "job_title_normalized", "job_summary_normalized", "skills"],
    )
    resumes = spark.createDataFrame(
        [("resume-1", "data scientist", "python experience", "SECRET JD", 1)],
        ["resume_id", "role_normalized", "resume_normalized", "Job_Description", "label"],
    )

    job_doc = build_job_documents(jobs).first()
    resume_doc = build_resume_documents(resumes).first()

    assert job_doc.combined_text == "data scientist build models python spark"
    assert resume_doc.combined_text == "data scientist python experience"
    assert "SECRET" not in resume_doc.combined_text
    assert resume_doc.document_type == "resume"


def test_select_resume_queries_uses_only_requested_split(spark) -> None:
    resumes = spark.createDataFrame(
        [
            ("r3", "validation"),
            ("r1", "test"),
            ("r2", "validation"),
        ],
        ["resume_id", "split"],
    )

    selected = select_resume_queries(resumes, "validation", limit=1).collect()

    assert [row.resume_id for row in selected] == ["r2"]
