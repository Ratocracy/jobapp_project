"""Exact dense cosine retrieval for normalized transformer embeddings."""

from __future__ import annotations

import numpy as np
from pyspark.sql import DataFrame, SparkSession

from jobapps.validation import require_columns


def exact_dense_cosine_top_k(
    spark: SparkSession,
    resume_embeddings: DataFrame,
    job_embeddings: DataFrame,
    top_k: int,
) -> DataFrame:
    """Rank all sampled jobs for each resume using normalized dense vectors."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    for frame in (resume_embeddings, job_embeddings):
        require_columns(frame, ["source_id", "embedding"])

    job_rows = job_embeddings.select("source_id", "embedding").orderBy(
        "source_id"
    ).collect()
    resume_rows = resume_embeddings.select("source_id", "embedding").orderBy(
        "source_id"
    ).collect()
    if not job_rows or not resume_rows:
        raise ValueError("Exact semantic retrieval requires jobs and resumes")
    if top_k > len(job_rows):
        raise ValueError("top_k cannot exceed the number of jobs")

    job_matrix = np.asarray([row["embedding"] for row in job_rows], dtype=np.float32)
    resume_matrix = np.asarray(
        [row["embedding"] for row in resume_rows], dtype=np.float32
    )
    if job_matrix.ndim != 2 or resume_matrix.ndim != 2:
        raise ValueError("Embeddings must form two-dimensional matrices")
    if job_matrix.shape[1] != resume_matrix.shape[1]:
        raise ValueError("Job and resume embedding dimensions must match")

    similarities = resume_matrix @ job_matrix.T
    job_ids = np.asarray([row["source_id"] for row in job_rows], dtype=object)
    results: list[tuple[str, int, str, float]] = []
    for resume_index, resume_row in enumerate(resume_rows):
        scores = similarities[resume_index]
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


def mean_top_k_overlap(left: DataFrame, right: DataFrame) -> float:
    """Return the mean fraction of left top-K pairs also present on the right."""

    for frame in (left, right):
        require_columns(frame, ["resume_id", "job_link"])
    left_pairs = {
        (row["resume_id"], row["job_link"])
        for row in left.select("resume_id", "job_link").collect()
    }
    right_pairs = {
        (row["resume_id"], row["job_link"])
        for row in right.select("resume_id", "job_link").collect()
    }
    return len(left_pairs & right_pairs) / len(left_pairs) if left_pairs else 0.0
