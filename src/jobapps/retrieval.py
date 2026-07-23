"""Approximate candidate generation and exact top-K reranking."""

from __future__ import annotations

from pyspark.ml.feature import BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, Window, functions as F

from jobapps.validation import require_columns


def fit_lsh_index(
    job_features: DataFrame,
    bucket_length: float = 0.8,
    num_hash_tables: int = 3,
    seed: int = 5110,
) -> BucketedRandomProjectionLSHModel:
    """Fit an LSH model to normalized job TF-IDF vectors."""

    require_columns(job_features, ["document_id", "source_id", "features"])
    estimator = BucketedRandomProjectionLSH(
        inputCol="features",
        outputCol="hashes",
        bucketLength=bucket_length,
        numHashTables=num_hash_tables,
        seed=seed,
    )
    return estimator.fit(job_features)


def generate_candidates(
    model: BucketedRandomProjectionLSHModel,
    resume_features: DataFrame,
    job_features: DataFrame,
    distance_threshold: float,
) -> DataFrame:
    """Generate approximate resume-job pairs within an Euclidean threshold."""

    for frame in (resume_features, job_features):
        require_columns(frame, ["document_id", "source_id", "features"])
    pairs = model.approxSimilarityJoin(
        resume_features,
        job_features,
        distance_threshold,
        distCol="approx_distance",
    )
    return pairs.select(
        F.col("datasetA.source_id").alias("resume_id"),
        F.col("datasetB.source_id").alias("job_link"),
        F.col("datasetA.features").alias("resume_features"),
        F.col("datasetB.features").alias("job_features"),
        "approx_distance",
    )


def rerank_top_k(candidates: DataFrame, top_k: int) -> DataFrame:
    """Compute exact cosine scores on normalized vectors and retain top K."""

    require_columns(
        candidates,
        ["resume_id", "job_link", "resume_features", "job_features", "approx_distance"],
    )
    scored = (
        candidates
        .withColumn("resume_array", vector_to_array("resume_features"))
        .withColumn("job_array", vector_to_array("job_features"))
        .withColumn(
            "similarity_score",
            F.expr(
                "aggregate(zip_with(resume_array, job_array, (x, y) -> x * y), "
                "cast(0.0 as double), (acc, value) -> acc + value)"
            ),
        )
    )
    order = Window.partitionBy("resume_id").orderBy(
        F.desc("similarity_score"), F.asc("approx_distance"), F.asc("job_link")
    )
    return (
        scored
        .withColumn("rank", F.row_number().over(order))
        .filter(F.col("rank") <= top_k)
        .select(
            "resume_id",
            "rank",
            "job_link",
            "similarity_score",
            "approx_distance",
        )
    )
