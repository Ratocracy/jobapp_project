from jobapps.schemas import RESUME_SCREENING_SCHEMA
from jobapps.transforms.resumes import build_silver_resumes


def test_resume_silver_masks_direct_pii_and_assigns_group_split(spark) -> None:
    source = spark.createDataFrame(
        [
            (
                "Data Scientist",
                "Here's a professional resume for Jane Doe:\n"
                "Jane Doe\nEmail jane@example.com Phone (617) 555-1234",
                "select",
                "Strong skills",
                "Build Python models",
            ),
            (
                "Analyst",
                "Contact john@example.com or 212-555-9999",
                "reject",
                "Missing experience",
                "Build Python models",
            ),
        ],
        RESUME_SCREENING_SCHEMA,
    )

    rows = build_silver_resumes(source).collect()

    assert len({row.resume_id for row in rows}) == 2
    assert len({row.job_description_id for row in rows}) == 1
    assert len({row.split for row in rows}) == 1
    assert "jane@example.com" not in rows[0].resume_normalized
    assert "[email]" in rows[0].resume_normalized
    assert "[phone]" in rows[0].resume_normalized
    assert "jane doe" not in rows[0].resume_normalized.split("professional resume for", 1)[1]
    assert rows[0].label == 1
    assert rows[1].label == 0
