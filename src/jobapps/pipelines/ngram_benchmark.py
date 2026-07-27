"""Benchmark Spark n-gram TF-IDF against unigram TF-IDF on validation queries."""

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
from jobapps.ngram_features import (
    fit_ngram_tfidf_pipeline,
    transform_ngram_documents,
)
from jobapps.pipelines.jobs_pipeline import create_spark_session, run_jobs_pipeline
from jobapps.pipelines.resumes_pipeline import run_resumes_pipeline
from jobapps.retrieval_benchmark import exact_cosine_top_k
from jobapps.semantic_retrieval import mean_top_k_overlap


@dataclass(frozen=True)
class NgramBenchmarkMetrics:
    job_documents: int
    resume_queries: int
    unigram_seconds: float
    ngram_seconds: float
    mean_top_k_overlap: float
    total_seconds: float


def run_ngram_benchmark(
    config_path: str | Path,
    quality_path: str | Path,
) -> NgramBenchmarkMetrics:
    config = load_pipeline_config(config_path)
    if config.runtime.resume_query_split == "test":
        raise ValueError("Development benchmark cannot consume the locked test split")
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

        unigram_started = perf_counter()
        unigram_model = fit_tfidf_pipeline(
            job_documents,
            config.nlp.num_features,
            config.nlp.min_document_frequency,
        )
        unigram_top_k = exact_cosine_top_k(
            spark,
            transform_documents(unigram_model, resume_documents),
            transform_documents(unigram_model, job_documents),
            config.retrieval.top_k,
        ).cache()
        unigram_top_k.count()
        unigram_seconds = perf_counter() - unigram_started

        ngram_started = perf_counter()
        ngram_model = fit_ngram_tfidf_pipeline(
            job_documents,
            config.ngram.num_features,
            config.ngram.min_document_frequency,
        )
        ngram_top_k = exact_cosine_top_k(
            spark,
            transform_ngram_documents(ngram_model, resume_documents),
            transform_ngram_documents(ngram_model, job_documents),
            config.retrieval.top_k,
        ).cache()
        ngram_top_k.count()
        ngram_seconds = perf_counter() - ngram_started

        metrics = NgramBenchmarkMetrics(
            job_documents=job_count,
            resume_queries=resume_count,
            unigram_seconds=round(unigram_seconds, 3),
            ngram_seconds=round(ngram_seconds, 3),
            mean_top_k_overlap=round(
                mean_top_k_overlap(unigram_top_k, ngram_top_k),
                6,
            ),
            total_seconds=round(perf_counter() - started, 3),
        )
        output_dir = (
            config.output.gold_dir
            / "benchmarks"
            / f"ngram_1pct_{config.runtime.resume_query_split}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        unigram_top_k.write.mode("overwrite").parquet(
            str(output_dir / "unigram_top_k")
        )
        ngram_top_k.write.mode("overwrite").parquet(
            str(output_dir / "ngram_top_k")
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    **asdict(metrics),
                    "resume_query_split": config.runtime.resume_query_split,
                    "ngram_num_features": config.ngram.num_features,
                    "top_k": config.retrieval.top_k,
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
    parser.add_argument("--config", default="config/ngram_1pct.yaml")
    parser.add_argument("--quality-config", default="config/data_quality.yaml")
    args = parser.parse_args()
    print(asdict(run_ngram_benchmark(args.config, args.quality_config)))


if __name__ == "__main__":
    main()
