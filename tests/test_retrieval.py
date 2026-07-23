import pytest

from jobapps.features import fit_tfidf_pipeline, transform_documents
from jobapps.retrieval import fit_lsh_index, generate_candidates, rerank_top_k


def test_tfidf_lsh_returns_matching_job_first(spark) -> None:
    jobs = spark.createDataFrame(
        [
            ("job-doc-1", "job", "job-1", "python spark machine learning models"),
            ("job-doc-2", "job", "job-2", "clinical nursing patient hospital care"),
        ],
        ["document_id", "document_type", "source_id", "combined_text"],
    )
    resumes = spark.createDataFrame(
        [("resume-doc-1", "resume", "resume-1", "python spark machine learning")],
        ["document_id", "document_type", "source_id", "combined_text"],
    )

    feature_model = fit_tfidf_pipeline(jobs, num_features=1024, min_document_frequency=1)
    job_features = transform_documents(feature_model, jobs)
    resume_features = transform_documents(feature_model, resumes)
    lsh_model = fit_lsh_index(job_features, bucket_length=1.0, num_hash_tables=4)
    candidates = generate_candidates(lsh_model, resume_features, job_features, 2.0)
    result = rerank_top_k(candidates, top_k=1).first()

    assert result.resume_id == "resume-1"
    assert result.job_link == "job-1"
    assert result.rank == 1
    assert result.similarity_score > 0.5


def test_rerank_converts_normalized_euclidean_distance_to_cosine(spark) -> None:
    candidates = spark.createDataFrame(
        [
            ("resume-1", "job-1", None, None, 0.0),
            ("resume-1", "job-2", None, None, 1.0),
        ],
        "resume_id string, job_link string, resume_features string, "
        "job_features string, approx_distance double",
    )

    results = rerank_top_k(candidates, top_k=2).collect()

    assert [row.job_link for row in results] == ["job-1", "job-2"]
    assert results[0].similarity_score == pytest.approx(1.0)
    assert results[1].similarity_score == pytest.approx(0.5)
