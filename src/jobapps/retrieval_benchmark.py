"""Exact sparse retrieval and recall metrics for sampled benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql import DataFrame, SparkSession, functions as F
from scipy.sparse import csr_matrix

from jobapps.validation import require_columns


@dataclass(frozen=True)
class RecallMetrics:
    candidate_recall_at_k: float
    recommendation_recall_at_k: float


def _rows_to_csr(rows: list, vector_column: str) -> csr_matrix:
    """Convert collected Spark vectors to a SciPy CSR matrix without densifying."""

    if not rows:
        raise ValueError("Cannot build an exact-retrieval matrix from zero rows")

    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    vector_size: int | None = None

    for row in rows:
        vector = row[vector_column]
        if not isinstance(vector, (SparseVector, DenseVector)):
            raise TypeError(f"{vector_column} must contain Spark ML vectors")
        if vector_size is None:
            vector_size = vector.size
        elif vector.size != vector_size:
            raise ValueError("All feature vectors must have the same size")

        if isinstance(vector, SparseVector):
            indices.extend(int(index) for index in vector.indices)
            data.extend(float(value) for value in vector.values)
        else:
            nonzero = np.flatnonzero(vector.values)
            indices.extend(int(index) for index in nonzero)
            data.extend(float(vector[index]) for index in nonzero)
        indptr.append(len(data))

    return csr_matrix(
        (data, indices, indptr),
        shape=(len(rows), int(vector_size)),
        dtype=np.float64,
    )


def exact_cosine_top_k(
    spark: SparkSession,
    resume_features: DataFrame,
    job_features: DataFrame,
    top_k: int,
) -> DataFrame:
    """Exhaustively rank a sampled catalog using normalized sparse vectors."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    for frame in (resume_features, job_features):
        require_columns(frame, ["source_id", "features"])

    job_rows = (
        job_features.select("source_id", "features").orderBy("source_id").collect()
    )
    resume_rows = (
        resume_features.select("source_id", "features").orderBy("source_id").collect()
    )
    if top_k > len(job_rows):
        raise ValueError("top_k cannot exceed the number of jobs")

    job_matrix = _rows_to_csr(job_rows, "features")
    resume_matrix = _rows_to_csr(resume_rows, "features")
    similarities = resume_matrix @ job_matrix.transpose()
    job_ids = np.asarray([row["source_id"] for row in job_rows], dtype=object)

    results: list[tuple[str, int, str, float]] = []
    for resume_index, resume_row in enumerate(resume_rows):
        scores = similarities.getrow(resume_index).toarray().ravel()
        ranked_indices = np.lexsort((job_ids, -scores))[:top_k]
        results.extend(
            (
                resume_row["source_id"],
                rank,
                str(job_ids[job_index]),
                float(scores[job_index]),
            )
            for rank, job_index in enumerate(ranked_indices, start=1)
        )

    return spark.createDataFrame(
        results,
        "resume_id string, rank int, job_link string, similarity_score double",
    )


def measure_lsh_recall(
    exact_top_k: DataFrame,
    candidates: DataFrame,
    recommendations: DataFrame,
) -> RecallMetrics:
    """Measure candidate and final recommendation overlap with exact top-K."""

    for frame in (exact_top_k, candidates, recommendations):
        require_columns(frame, ["resume_id", "job_link"])

    exact_pairs = exact_top_k.select("resume_id", "job_link").distinct()

    def overlap(frame: DataFrame) -> float:
        approximate_pairs = frame.select("resume_id", "job_link").distinct()
        matched = exact_pairs.join(
            approximate_pairs.withColumn("found", F.lit(1)),
            on=["resume_id", "job_link"],
            how="left",
        )
        value = matched.agg(
            F.avg(F.coalesce(F.col("found"), F.lit(0))).alias("recall")
        ).first()["recall"]
        return float(value or 0.0)

    return RecallMetrics(
        candidate_recall_at_k=overlap(candidates),
        recommendation_recall_at_k=overlap(recommendations),
    )
