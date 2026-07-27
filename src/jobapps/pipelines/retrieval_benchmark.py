"""Benchmark 1% LSH retrieval against exhaustive sparse cosine top-K."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

from jobapps.config import load_pipeline_config
from jobapps.documents import (
    build_job_documents,
    build_resume_documents,
    select_resume_queries,
)
from jobapps.features import fit_tfidf_pipeline, transform_documents
from jobapps.pipelines.jobs_pipeline import create_spark_session, run_jobs_pipeline
from jobapps.pipelines.resumes_pipeline import run_resumes_pipeline
from jobapps.retrieval import fit_lsh_index, generate_candidates, rerank_top_k
from jobapps.retrieval_benchmark import exact_cosine_top_k, measure_lsh_recall


@dataclass(frozen=True)
class BenchmarkMetrics:
    job_documents: int
    resume_queries: int
    exhaustive_pairs: int
    candidate_pairs: int
    matched_resumes: int
    candidate_recall_at_k: float
    recommendation_recall_at_k: float
    exact_seconds: float
    lsh_seconds: float
    total_seconds: float


def run_retrieval_benchmark(config_path: str | Path, quality_path: str | Path):
    config = load_pipeline_config(config_path)
    if config.runtime.sample_fraction > 0.01:
        raise ValueError("Exact benchmark is restricted to a sample fraction of at most 1%")

    spark = create_spark_session(config)
    started = perf_counter()
    try:
        silver_jobs, _, _ = run_jobs_pipeline(
            spark, config_path, quality_path, write_output=False
        )
        silver_resumes, _, _ = run_resumes_pipeline(
            spark, config, quality_path, write_output=False
        )
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

        feature_model = fit_tfidf_pipeline(
            job_documents,
            num_features=config.nlp.num_features,
            min_document_frequency=config.nlp.min_document_frequency,
        )
        job_features = transform_documents(feature_model, job_documents).cache()
        resume_features = transform_documents(feature_model, resume_documents).cache()
        job_features.count()
        resume_features.count()

        exact_started = perf_counter()
        exact_results = exact_cosine_top_k(
            spark, resume_features, job_features, config.retrieval.top_k
        ).cache()
        exact_results.count()
        exact_seconds = perf_counter() - exact_started

        lsh_started = perf_counter()
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
            config.retrieval.distance_threshold,
        ).cache()
        candidate_count = candidates.count()
        recommendations = rerank_top_k(
            candidates, config.retrieval.top_k
        ).cache()
        recommendations.count()
        matched_resumes = recommendations.select("resume_id").distinct().count()
        recall = measure_lsh_recall(exact_results, candidates, recommendations)
        lsh_seconds = perf_counter() - lsh_started

        metrics = BenchmarkMetrics(
            job_documents=job_count,
            resume_queries=resume_count,
            exhaustive_pairs=job_count * resume_count,
            candidate_pairs=candidate_count,
            matched_resumes=matched_resumes,
            candidate_recall_at_k=round(recall.candidate_recall_at_k, 6),
            recommendation_recall_at_k=round(
                recall.recommendation_recall_at_k, 6
            ),
            exact_seconds=round(exact_seconds, 3),
            lsh_seconds=round(lsh_seconds, 3),
            total_seconds=round(perf_counter() - started, 3),
        )

        output_dir = (
            config.output.gold_dir
            / "benchmarks"
            / f"exact_1pct_{config.runtime.resume_query_split}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        exact_results.write.mode("overwrite").parquet(
            str(output_dir / "exact_top_k")
        )
        recommendations.write.mode("overwrite").parquet(
            str(output_dir / "lsh_top_k")
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    **asdict(metrics),
                    "sample_fraction": config.runtime.sample_fraction,
                    "resume_query_split": config.runtime.resume_query_split,
                    "top_k": config.retrieval.top_k,
                    "bucket_length": config.retrieval.bucket_length,
                    "num_hash_tables": config.retrieval.num_hash_tables,
                    "distance_threshold": config.retrieval.distance_threshold,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return metrics
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/benchmark_1pct.yaml")
    parser.add_argument("--quality-config", default="config/data_quality.yaml")
    args = parser.parse_args()
    metrics = run_retrieval_benchmark(args.config, args.quality_config)
    print(asdict(metrics))


if __name__ == "__main__":
    main()
