import pytest
from pyspark.ml.linalg import Vectors

from jobapps.retrieval_benchmark import exact_cosine_top_k, measure_lsh_recall


def test_exact_cosine_top_k_ranks_normalized_sparse_vectors(spark) -> None:
    jobs = spark.createDataFrame(
        [
            ("job-b", Vectors.sparse(3, {1: 1.0})),
            ("job-a", Vectors.sparse(3, {0: 1.0})),
        ],
        ["source_id", "features"],
    )
    resumes = spark.createDataFrame(
        [("resume-1", Vectors.sparse(3, {0: 1.0}))],
        ["source_id", "features"],
    )

    results = exact_cosine_top_k(spark, resumes, jobs, top_k=2).collect()

    assert [row.job_link for row in results] == ["job-a", "job-b"]
    assert results[0].similarity_score == pytest.approx(1.0)
    assert results[1].similarity_score == pytest.approx(0.0)


def test_measure_lsh_recall_distinguishes_candidates_and_final_top_k(spark) -> None:
    exact = spark.createDataFrame(
        [("resume-1", "job-a"), ("resume-1", "job-b")],
        ["resume_id", "job_link"],
    )
    candidates = spark.createDataFrame(
        [("resume-1", "job-a"), ("resume-1", "job-b")],
        ["resume_id", "job_link"],
    )
    recommendations = spark.createDataFrame(
        [("resume-1", "job-a"), ("resume-1", "job-c")],
        ["resume_id", "job_link"],
    )

    metrics = measure_lsh_recall(exact, candidates, recommendations)

    assert metrics.candidate_recall_at_k == pytest.approx(1.0)
    assert metrics.recommendation_recall_at_k == pytest.approx(0.5)
