"""Compare exact TF-IDF and transformer cosine retrieval on the 1% sample."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

from pyspark.sql import DataFrame, functions as F

from jobapps.config import PipelineConfig, load_pipeline_config
from jobapps.documents import build_job_documents, build_resume_documents
from jobapps.features import fit_tfidf_pipeline, transform_documents
from jobapps.pipelines.jobs_pipeline import create_spark_session, run_jobs_pipeline
from jobapps.pipelines.resumes_pipeline import run_resumes_pipeline
from jobapps.retrieval_benchmark import exact_cosine_top_k
from jobapps.semantic_retrieval import exact_dense_cosine_top_k, mean_top_k_overlap
from jobapps.transformer_features import embed_documents, load_sentence_transformer


@dataclass(frozen=True)
class TransformerBenchmarkMetrics:
    job_documents: int
    resume_queries: int
    embedding_dimension: int
    job_chunks: int
    resume_chunks: int
    model_load_seconds: float
    embedding_seconds: float
    transformer_retrieval_seconds: float
    tfidf_seconds: float
    mean_top_k_overlap: float
    total_seconds: float


def _write_frame(frame: DataFrame, path: Path, partitions: int) -> None:
    frame.coalesce(partitions).write.mode("overwrite").parquet(str(path))


def _cached_embeddings_are_valid(
    frame: DataFrame,
    expected_count: int,
    expected_model: str,
) -> bool:
    row = frame.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("embedding_model").alias("models"),
        F.first("embedding_model").alias("model"),
    ).first()
    return (
        int(row["rows"]) == expected_count
        and int(row["models"]) == 1
        and row["model"] == expected_model
    )


def _create_embeddings(
    spark,
    config: PipelineConfig,
    job_documents: DataFrame,
    resume_documents: DataFrame,
    output_dir: Path,
    force_recompute: bool,
) -> tuple[DataFrame, DataFrame, float, float]:
    job_path = output_dir / "job_embeddings"
    resume_path = output_dir / "resume_embeddings"
    job_count = job_documents.count()
    resume_count = resume_documents.count()

    if not force_recompute and job_path.exists() and resume_path.exists():
        cached_jobs = spark.read.parquet(str(job_path))
        cached_resumes = spark.read.parquet(str(resume_path))
        if _cached_embeddings_are_valid(
            cached_jobs, job_count, config.transformer.model_name
        ) and _cached_embeddings_are_valid(
            cached_resumes, resume_count, config.transformer.model_name
        ):
            return cached_jobs, cached_resumes, 0.0, 0.0

    model_started = perf_counter()
    model = load_sentence_transformer(
        config.transformer.model_name,
        config.transformer.device,
    )
    model_load_seconds = perf_counter() - model_started

    embedding_started = perf_counter()
    job_embeddings = embed_documents(
        spark,
        job_documents,
        model=model,
        model_name=config.transformer.model_name,
        batch_size=config.transformer.batch_size,
        max_tokens=config.transformer.max_tokens,
        overlap_tokens=config.transformer.overlap_tokens,
    ).cache()
    resume_embeddings = embed_documents(
        spark,
        resume_documents,
        model=model,
        model_name=config.transformer.model_name,
        batch_size=config.transformer.batch_size,
        max_tokens=config.transformer.max_tokens,
        overlap_tokens=config.transformer.overlap_tokens,
    ).cache()
    job_embeddings.count()
    resume_embeddings.count()
    _write_frame(
        job_embeddings,
        job_path,
        config.runtime.output_partitions,
    )
    _write_frame(
        resume_embeddings,
        resume_path,
        config.runtime.output_partitions,
    )
    return (
        job_embeddings,
        resume_embeddings,
        model_load_seconds,
        perf_counter() - embedding_started,
    )


def run_transformer_benchmark(
    config_path: str | Path,
    quality_path: str | Path,
    force_recompute: bool = False,
) -> TransformerBenchmarkMetrics:
    config = load_pipeline_config(config_path)
    if config.runtime.sample_fraction != 0.01:
        raise ValueError("Transformer benchmark requires the controlled 1% sample")

    spark = create_spark_session(config)
    started = perf_counter()
    output_dir = config.output.gold_dir / "benchmarks" / "transformer_1pct"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        silver_jobs, _, _ = run_jobs_pipeline(
            spark, config_path, quality_path, write_output=False
        )
        silver_resumes, _, _ = run_resumes_pipeline(
            spark, config, quality_path, write_output=False
        )
        job_documents = build_job_documents(silver_jobs).cache()
        resume_documents = (
            build_resume_documents(silver_resumes)
            .orderBy("document_id")
            .limit(config.runtime.resume_query_limit)
            .cache()
        )
        job_count = job_documents.count()
        resume_count = resume_documents.count()

        job_embeddings, resume_embeddings, model_seconds, embedding_seconds = (
            _create_embeddings(
                spark,
                config,
                job_documents,
                resume_documents,
                output_dir,
                force_recompute,
            )
        )

        transformer_started = perf_counter()
        transformer_top_k = exact_dense_cosine_top_k(
            spark,
            resume_embeddings,
            job_embeddings,
            config.retrieval.top_k,
        ).cache()
        transformer_top_k.count()
        transformer_seconds = perf_counter() - transformer_started

        tfidf_started = perf_counter()
        tfidf_model = fit_tfidf_pipeline(
            job_documents,
            config.nlp.num_features,
            config.nlp.min_document_frequency,
        )
        job_tfidf = transform_documents(tfidf_model, job_documents)
        resume_tfidf = transform_documents(tfidf_model, resume_documents)
        tfidf_top_k = exact_cosine_top_k(
            spark,
            resume_tfidf,
            job_tfidf,
            config.retrieval.top_k,
        ).cache()
        tfidf_top_k.count()
        tfidf_seconds = perf_counter() - tfidf_started
        overlap = mean_top_k_overlap(transformer_top_k, tfidf_top_k)

        embedding_dimension = int(
            job_embeddings.select(F.size("embedding").alias("size")).first()["size"]
        )
        job_chunks = int(
            job_embeddings.agg(F.sum("chunk_count").alias("chunks")).first()["chunks"]
        )
        resume_chunks = int(
            resume_embeddings.agg(F.sum("chunk_count").alias("chunks")).first()[
                "chunks"
            ]
        )
        metrics = TransformerBenchmarkMetrics(
            job_documents=job_count,
            resume_queries=resume_count,
            embedding_dimension=embedding_dimension,
            job_chunks=job_chunks,
            resume_chunks=resume_chunks,
            model_load_seconds=round(model_seconds, 3),
            embedding_seconds=round(embedding_seconds, 3),
            transformer_retrieval_seconds=round(transformer_seconds, 3),
            tfidf_seconds=round(tfidf_seconds, 3),
            mean_top_k_overlap=round(overlap, 6),
            total_seconds=round(perf_counter() - started, 3),
        )
        _write_frame(
            transformer_top_k,
            output_dir / "transformer_top_k",
            config.runtime.output_partitions,
        )
        _write_frame(
            tfidf_top_k,
            output_dir / "tfidf_top_k",
            config.runtime.output_partitions,
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    **asdict(metrics),
                    "model_name": config.transformer.model_name,
                    "batch_size": config.transformer.batch_size,
                    "max_tokens": config.transformer.max_tokens,
                    "overlap_tokens": config.transformer.overlap_tokens,
                    "device": config.transformer.device,
                    "sample_fraction": config.runtime.sample_fraction,
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
    parser.add_argument("--config", default="config/transformer_1pct.yaml")
    parser.add_argument("--quality-config", default="config/data_quality.yaml")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    metrics = run_transformer_benchmark(
        args.config,
        args.quality_config,
        force_recompute=args.force_recompute,
    )
    print(asdict(metrics))


if __name__ == "__main__":
    main()
