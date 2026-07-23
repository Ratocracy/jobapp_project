"""Leakage-safe proxy evaluation using labeled resume/job-description pairs."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F

from jobapps.validation import require_columns


@dataclass(frozen=True)
class RetrievalEvaluation:
    evaluated_pairs: int
    roc_auc: float
    pr_auc: float


def build_evaluation_documents(silver_resumes: DataFrame) -> DataFrame:
    """Create paired job-description documents for evaluation only."""

    require_columns(
        silver_resumes,
        [
            "resume_id",
            "job_description_id",
            "job_description_normalized",
            "label",
            "split",
        ],
    )
    pair_id = F.sha2(
        F.concat_ws(":", F.col("resume_id"), F.col("job_description_id")), 256
    )
    return silver_resumes.select(
        pair_id.alias("document_id"),
        F.lit("evaluation_job_description").alias("document_type"),
        pair_id.alias("source_id"),
        F.col("job_description_normalized").alias("combined_text"),
        "resume_id",
        "label",
        "split",
    )


def score_labeled_pairs(
    resume_features: DataFrame,
    evaluation_features: DataFrame,
) -> DataFrame:
    """Calculate exact cosine similarity for known resume-description pairs."""

    require_columns(resume_features, ["source_id", "features"])
    require_columns(evaluation_features, ["resume_id", "label", "split", "features"])
    resumes = resume_features.select(
        F.col("source_id").alias("resume_id"),
        F.col("features").alias("resume_features"),
    )
    pairs = evaluation_features.select(
        "document_id",
        "resume_id",
        "label",
        "split",
        F.col("features").alias("job_description_features"),
    )
    return (
        pairs.join(resumes, on="resume_id", how="inner")
        .withColumn("resume_array", vector_to_array("resume_features"))
        .withColumn(
            "job_description_array", vector_to_array("job_description_features")
        )
        .withColumn(
            "similarity_score",
            F.expr(
                "aggregate(zip_with(resume_array, job_description_array, "
                "(x, y) -> x * y), cast(0.0 as double), "
                "(acc, value) -> acc + value)"
            ),
        )
        .select("document_id", "resume_id", "label", "split", "similarity_score")
    )


def evaluate_labeled_pairs(scores: DataFrame) -> RetrievalEvaluation:
    """Measure whether similarity separates selected from rejected pairs."""

    require_columns(scores, ["label", "similarity_score"])
    pair_count = scores.count()
    roc_auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="similarity_score",
        metricName="areaUnderROC",
    ).evaluate(scores)
    pr_auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="similarity_score",
        metricName="areaUnderPR",
    ).evaluate(scores)
    return RetrievalEvaluation(
        evaluated_pairs=pair_count,
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
    )
