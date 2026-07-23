"""Unsupervised TF-IDF feature pipeline shared by jobs and resumes."""

from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import HashingTF, IDF, Normalizer, RegexTokenizer, StopWordsRemover
from pyspark.sql import DataFrame

from jobapps.documents import require_document_contract


def build_tfidf_pipeline(
    num_features: int = 262_144,
    min_document_frequency: int = 2,
) -> Pipeline:
    """Create an unsupervised Spark ML pipeline for the document contract."""

    tokenizer = RegexTokenizer(
        inputCol="combined_text",
        outputCol="tokens_raw",
        pattern=r"[^\p{L}\p{N}+#.]+",
        gaps=True,
        minTokenLength=2,
        toLowercase=True,
    )
    stop_words = StopWordsRemover(
        inputCol="tokens_raw",
        outputCol="tokens",
        caseSensitive=False,
    )
    term_frequency = HashingTF(
        inputCol="tokens",
        outputCol="term_frequency",
        numFeatures=num_features,
        binary=False,
    )
    inverse_document_frequency = IDF(
        inputCol="term_frequency",
        outputCol="tfidf",
        minDocFreq=min_document_frequency,
    )
    normalizer = Normalizer(inputCol="tfidf", outputCol="features", p=2.0)
    return Pipeline(
        stages=[
            tokenizer,
            stop_words,
            term_frequency,
            inverse_document_frequency,
            normalizer,
        ]
    )


def fit_tfidf_pipeline(
    job_documents: DataFrame,
    num_features: int = 262_144,
    min_document_frequency: int = 2,
) -> PipelineModel:
    """Fit corpus statistics on jobs only."""

    require_document_contract(job_documents)
    return build_tfidf_pipeline(num_features, min_document_frequency).fit(job_documents)


def transform_documents(model: PipelineModel, documents: DataFrame) -> DataFrame:
    """Apply one fitted feature model to either side of the retrieval system."""

    require_document_contract(documents)
    return model.transform(documents)
