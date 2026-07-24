import pytest

from jobapps.semantic_retrieval import exact_dense_cosine_top_k, mean_top_k_overlap


def test_exact_dense_cosine_top_k_ranks_semantic_vectors(spark) -> None:
    jobs = spark.createDataFrame(
        [("job-b", [0.0, 1.0]), ("job-a", [1.0, 0.0])],
        "source_id string, embedding array<double>",
    )
    resumes = spark.createDataFrame(
        [("resume-1", [1.0, 0.0])],
        "source_id string, embedding array<double>",
    )

    results = exact_dense_cosine_top_k(spark, resumes, jobs, top_k=2).collect()

    assert [row.job_link for row in results] == ["job-a", "job-b"]
    assert results[0].similarity_score == pytest.approx(1.0)


def test_mean_top_k_overlap(spark) -> None:
    left = spark.createDataFrame(
        [("resume-1", "job-a"), ("resume-1", "job-b")],
        ["resume_id", "job_link"],
    )
    right = spark.createDataFrame(
        [("resume-1", "job-b"), ("resume-1", "job-c")],
        ["resume_id", "job_link"],
    )

    assert mean_top_k_overlap(left, right) == pytest.approx(0.5)
