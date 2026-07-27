"""Run the Week 1-3 retrieval MVP from raw sources through top-K matches."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

from pyspark.sql import SparkSession, functions as F

from jobapps.config import PipelineConfig, load_pipeline_config
from jobapps.documents import (
    build_job_documents,
    build_resume_documents,
    select_resume_queries,
)
from jobapps.evaluation import (
    build_evaluation_documents,
    evaluate_labeled_pairs,
    score_labeled_pairs,
)
from jobapps.features import fit_tfidf_pipeline, transform_documents
from jobapps.pipelines.jobs_pipeline import create_spark_session, run_jobs_pipeline
from jobapps.pipelines.resumes_pipeline import run_resumes_pipeline
from jobapps.retrieval import fit_lsh_index, generate_candidates, rerank_top_k


@dataclass(frozen=True)
class MvpMetrics:
    job_documents: int
    resume_queries: int
    candidate_pairs: int
    matched_resumes: int
    recommendation_rows: int
    evaluated_pairs: int
    roc_auc: float
    pr_auc: float
    silver_seconds: float
    feature_seconds: float
    retrieval_seconds: float
    total_seconds: float


def _write_frame(dataframe, path: Path, partitions: int) -> None:
    dataframe.coalesce(partitions).write.mode("overwrite").parquet(str(path))


def run_mvp_pipeline(
    spark: SparkSession,
    config: PipelineConfig,
    config_path: str | Path,
    quality_path: str | Path,
    write_output: bool = True,
):
    started = perf_counter()

    silver_started = perf_counter()
    silver_jobs, _, _ = run_jobs_pipeline(
        spark, config_path, quality_path, write_output=write_output
    )
    silver_resumes, _, _ = run_resumes_pipeline(
        spark, config, quality_path, write_output=write_output
    )
    silver_seconds = perf_counter() - silver_started

    job_documents = build_job_documents(silver_jobs).cache()
    resume_documents = build_resume_documents(
        select_resume_queries(
            silver_resumes,
            config.runtime.resume_query_split,
            config.runtime.resume_query_limit,
        )
    ).cache()
    job_count = job_documents.count()
    resume_count = resume_documents.count()

    feature_started = perf_counter()
    feature_model = fit_tfidf_pipeline(
        job_documents,
        num_features=config.nlp.num_features,
        min_document_frequency=config.nlp.min_document_frequency,
    )
    job_features = transform_documents(feature_model, job_documents).cache()
    resume_features = transform_documents(feature_model, resume_documents).cache()
    job_features.count()
    resume_features.count()
    feature_seconds = perf_counter() - feature_started

    retrieval_started = perf_counter()
    lsh_model = fit_lsh_index(
        job_features,
        bucket_length=config.retrieval.bucket_length,
        num_hash_tables=config.retrieval.num_hash_tables,
        seed=config.runtime.random_seed,
    )
    candidates = generate_candidates(
        lsh_model,
        resume_features,
        job_features,
        distance_threshold=config.retrieval.distance_threshold,
    ).cache()
    candidate_count = candidates.count()
    recommendations = rerank_top_k(candidates, config.retrieval.top_k).cache()
    recommendation_count = recommendations.count()
    matched_resumes = recommendations.select("resume_id").distinct().count()

    query_ids = resume_documents.select(F.col("source_id").alias("resume_id"))
    evaluation_documents = build_evaluation_documents(
        silver_resumes.join(query_ids, on="resume_id", how="left_semi")
    )
    evaluation_features = transform_documents(feature_model, evaluation_documents)
    evaluation_scores = score_labeled_pairs(
        resume_features, evaluation_features
    ).cache()
    evaluation = evaluate_labeled_pairs(evaluation_scores)
    retrieval_seconds = perf_counter() - retrieval_started

    if write_output:
        _write_frame(
            job_features,
            config.output.gold_dir / "job_features",
            config.runtime.output_partitions,
        )
        _write_frame(
            resume_features,
            config.output.gold_dir / "resume_features",
            config.runtime.output_partitions,
        )
        _write_frame(
            recommendations,
            config.output.gold_dir / "top_k_recommendations",
            config.runtime.output_partitions,
        )
        _write_frame(
            evaluation_scores,
            config.output.gold_dir / "evaluation_scores",
            config.runtime.output_partitions,
        )
        feature_model.write().overwrite().save(
            str(config.output.gold_dir / "models" / "tfidf")
        )
        lsh_model.write().overwrite().save(
            str(config.output.gold_dir / "models" / "lsh")
        )

    metrics = MvpMetrics(
        job_documents=job_count,
        resume_queries=resume_count,
        candidate_pairs=candidate_count,
        matched_resumes=matched_resumes,
        recommendation_rows=recommendation_count,
        evaluated_pairs=evaluation.evaluated_pairs,
        roc_auc=round(evaluation.roc_auc, 6),
        pr_auc=round(evaluation.pr_auc, 6),
        silver_seconds=round(silver_seconds, 3),
        feature_seconds=round(feature_seconds, 3),
        retrieval_seconds=round(retrieval_seconds, 3),
        total_seconds=round(perf_counter() - started, 3),
    )
    if write_output:
        benchmark_path = config.output.gold_dir / "benchmark_metrics.json"
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_payload = {
            **asdict(metrics),
            "sample_fraction": config.runtime.sample_fraction,
            "resume_query_split": config.runtime.resume_query_split,
            "num_features": config.nlp.num_features,
            "distance_threshold": config.retrieval.distance_threshold,
            "top_k": config.retrieval.top_k,
        }
        benchmark_path.write_text(
            json.dumps(benchmark_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return recommendations, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--quality-config", default="config/data_quality.yaml")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    spark = create_spark_session(config)
    try:
        recommendations, metrics = run_mvp_pipeline(
            spark,
            config,
            args.config,
            args.quality_config,
            write_output=not args.no_write,
        )
        recommendations.show(20, truncate=60)
        print(asdict(metrics))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
