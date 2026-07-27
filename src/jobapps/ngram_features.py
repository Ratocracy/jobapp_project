"""Spark ML unigram-plus-bigram TF-IDF features for retrieval."""

from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    HashingTF,
    IDF,
    NGram,
    Normalizer,
    RegexTokenizer,
    SQLTransformer,
    StopWordsRemover,
)
from pyspark.sql import DataFrame

from jobapps.documents import require_document_contract


def build_ngram_tfidf_pipeline(
    num_features: int = 524_288,
    min_document_frequency: int = 2,
) -> Pipeline:
    """Build a Spark pipeline combining prefixed unigrams and bigrams."""

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
    bigrams = NGram(n=2, inputCol="tokens", outputCol="bigrams")
    combine = SQLTransformer(
        statement="""
        SELECT *,
          concat(
            transform(tokens, token -> concat('u_', token)),
            transform(
              bigrams,
              phrase -> concat('b_', replace(phrase, ' ', '_'))
            )
          ) AS ngram_tokens
        FROM __THIS__
        """
    )
    term_frequency = HashingTF(
        inputCol="ngram_tokens",
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
            bigrams,
            combine,
            term_frequency,
            inverse_document_frequency,
            normalizer,
        ]
    )


def fit_ngram_tfidf_pipeline(
    job_documents: DataFrame,
    num_features: int = 524_288,
    min_document_frequency: int = 2,
) -> PipelineModel:
    """Fit n-gram IDF corpus statistics on job documents only."""

    require_document_contract(job_documents)
    return build_ngram_tfidf_pipeline(
        num_features,
        min_document_frequency,
    ).fit(job_documents)


def transform_ngram_documents(
    model: PipelineModel,
    documents: DataFrame,
) -> DataFrame:
    """Transform jobs or resumes into the fitted n-gram feature space."""

    require_document_contract(documents)
    return model.transform(documents)
