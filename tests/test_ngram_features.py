from jobapps.ngram_features import (
    fit_ngram_tfidf_pipeline,
    transform_ngram_documents,
)


def test_ngram_pipeline_combines_prefixed_unigrams_and_bigrams(spark) -> None:
    documents = spark.createDataFrame(
        [
            ("d1", "job", "j1", "machine learning engineer"),
            ("d2", "job", "j2", "patient care nurse"),
        ],
        ["document_id", "document_type", "source_id", "combined_text"],
    )

    model = fit_ngram_tfidf_pipeline(
        documents,
        num_features=1024,
        min_document_frequency=1,
    )
    row = transform_ngram_documents(model, documents).orderBy("source_id").first()

    assert "u_machine" in row.ngram_tokens
    assert "b_machine_learning" in row.ngram_tokens
    assert row.features.numNonzeros() > 0
