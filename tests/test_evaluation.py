from jobapps.evaluation import evaluate_labeled_pairs, score_labeled_pairs
from jobapps.features import fit_tfidf_pipeline, transform_documents


def test_similarity_proxy_evaluation_separates_clear_pairs(spark) -> None:
    jobs = spark.createDataFrame(
        [
            ("job-1", "job", "job-1", "python spark machine learning"),
            ("job-2", "job", "job-2", "clinical nursing patient care"),
        ],
        ["document_id", "document_type", "source_id", "combined_text"],
    )
    resumes = spark.createDataFrame(
        [
            ("resume-1", "resume", "resume-1", "python spark machine learning"),
            ("resume-2", "resume", "resume-2", "python spark machine learning"),
        ],
        ["document_id", "document_type", "source_id", "combined_text"],
    )
    evaluation_documents = spark.createDataFrame(
        [
            ("pair-1", "evaluation_job_description", "pair-1", "python spark machine learning", "resume-1", 1, "test"),
            ("pair-2", "evaluation_job_description", "pair-2", "clinical nursing patient care", "resume-2", 0, "test"),
        ],
        ["document_id", "document_type", "source_id", "combined_text", "resume_id", "label", "split"],
    )

    model = fit_tfidf_pipeline(jobs, num_features=1024, min_document_frequency=1)
    resume_features = transform_documents(model, resumes)
    evaluation_features = transform_documents(model, evaluation_documents)
    scores = score_labeled_pairs(resume_features, evaluation_features)
    metrics = evaluate_labeled_pairs(scores)

    assert metrics.evaluated_pairs == 2
    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0
