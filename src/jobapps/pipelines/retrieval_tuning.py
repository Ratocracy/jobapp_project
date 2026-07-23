"""Tune LSH parameters against exact top-K on the controlled 1% benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

from jobapps.config import load_pipeline_config
from jobapps.documents import build_job_documents, build_resume_documents
from jobapps.features import fit_tfidf_pipeline, transform_documents
from jobapps.pipelines.jobs_pipeline import create_spark_session, run_jobs_pipeline
from jobapps.pipelines.resumes_pipeline import run_resumes_pipeline
from jobapps.retrieval import fit_lsh_index, generate_candidates
from jobapps.retrieval_benchmark import exact_cosine_top_k


BUCKET_LENGTHS = (0.5, 0.8, 1.0, 1.2, 1.5)
HASH_TABLE_COUNTS = (3, 5, 8, 12)
DISTANCE_THRESHOLDS = (1.2, 1.3, 1.4)


@dataclass(frozen=True)
class TuningResult:
    bucket_length: float
    num_hash_tables: int
    distance_threshold: float
    candidate_pairs: int
    resumes_covered: int
    candidate_recall_at_10: float
    final_recall_at_10: float
    runtime_seconds: float
    model_and_generation_seconds: float
    evaluation_seconds: float


def evaluate_candidate_rows(
    candidate_rows: list[tuple[str, str, float]],
    exact_pairs_by_resume: dict[str, set[str]],
    distance_threshold: float,
    top_k: int,
) -> tuple[int, int, float, float]:
    """Evaluate one threshold from candidates generated at the maximum threshold."""

    pairs: dict[tuple[str, str], float] = {}
    for resume_id, job_link, distance in candidate_rows:
        if distance <= distance_threshold:
            key = (resume_id, job_link)
            pairs[key] = min(distance, pairs.get(key, distance))

    by_resume: dict[str, list[tuple[str, float]]] = {}
    for (resume_id, job_link), distance in pairs.items():
        by_resume.setdefault(resume_id, []).append((job_link, distance))

    exact_pair_count = sum(len(jobs) for jobs in exact_pairs_by_resume.values())
    candidate_hits = 0
    final_hits = 0
    for resume_id, exact_jobs in exact_pairs_by_resume.items():
        candidates = by_resume.get(resume_id, [])
        candidate_jobs = {job_link for job_link, _ in candidates}
        candidate_hits += len(exact_jobs & candidate_jobs)
        ranked_jobs = {
            job_link
            for job_link, _ in sorted(candidates, key=lambda item: (item[1], item[0]))[
                :top_k
            ]
        }
        final_hits += len(exact_jobs & ranked_jobs)

    denominator = exact_pair_count or 1
    return (
        len(pairs),
        len(by_resume),
        candidate_hits / denominator,
        final_hits / denominator,
    )


def _write_results(results: list[TuningResult], output_dir: Path) -> None:
    """Checkpoint completed configurations in machine-readable formats."""

    if not results:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(result) for result in results]
    (output_dir / "results.json").write_text(
        json.dumps(records, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_tuning(config_path: str | Path, quality_path: str | Path) -> list[TuningResult]:
    config = load_pipeline_config(config_path)
    if config.runtime.sample_fraction != 0.01:
        raise ValueError("Retrieval tuning requires the controlled 1% configuration")
    if config.retrieval.top_k != 10:
        raise ValueError("Retrieval tuning currently reports recall@10 and requires top_k=10")

    spark = create_spark_session(config)
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
        job_documents.count()
        resume_documents.count()

        feature_model = fit_tfidf_pipeline(
            job_documents,
            config.nlp.num_features,
            config.nlp.min_document_frequency,
        )
        job_features = transform_documents(feature_model, job_documents).cache()
        resume_features = transform_documents(feature_model, resume_documents).cache()
        job_features.count()
        resume_features.count()

        exact_rows = exact_cosine_top_k(
            spark, resume_features, job_features, config.retrieval.top_k
        ).collect()
        exact_pairs_by_resume: dict[str, set[str]] = {}
        for row in exact_rows:
            exact_pairs_by_resume.setdefault(row["resume_id"], set()).add(
                row["job_link"]
            )

        results: list[TuningResult] = []
        output_dir = config.output.gold_dir / "benchmarks" / "lsh_tuning_1pct"
        maximum_threshold = max(DISTANCE_THRESHOLDS)
        for bucket_length in BUCKET_LENGTHS:
            for num_hash_tables in HASH_TABLE_COUNTS:
                generation_started = perf_counter()
                model = fit_lsh_index(
                    job_features,
                    bucket_length=bucket_length,
                    num_hash_tables=num_hash_tables,
                    seed=config.runtime.random_seed,
                )
                rows = (
                    generate_candidates(
                        model,
                        resume_features,
                        job_features,
                        maximum_threshold,
                    )
                    .select("resume_id", "job_link", "approx_distance")
                    .collect()
                )
                generation_seconds = perf_counter() - generation_started
                candidate_rows = [
                    (
                        row["resume_id"],
                        row["job_link"],
                        float(row["approx_distance"]),
                    )
                    for row in rows
                ]

                for threshold in DISTANCE_THRESHOLDS:
                    evaluation_started = perf_counter()
                    pair_count, coverage, candidate_recall, final_recall = (
                        evaluate_candidate_rows(
                            candidate_rows,
                            exact_pairs_by_resume,
                            threshold,
                            config.retrieval.top_k,
                        )
                    )
                    evaluation_seconds = perf_counter() - evaluation_started
                    results.append(
                        TuningResult(
                            bucket_length=bucket_length,
                            num_hash_tables=num_hash_tables,
                            distance_threshold=threshold,
                            candidate_pairs=pair_count,
                            resumes_covered=coverage,
                            candidate_recall_at_10=round(candidate_recall, 6),
                            final_recall_at_10=round(final_recall, 6),
                            runtime_seconds=round(
                                generation_seconds + evaluation_seconds, 3
                            ),
                            model_and_generation_seconds=round(
                                generation_seconds, 3
                            ),
                            evaluation_seconds=round(evaluation_seconds, 6),
                        )
                    )
                _write_results(results, output_dir)

        return results
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/benchmark_1pct.yaml")
    parser.add_argument("--quality-config", default="config/data_quality.yaml")
    args = parser.parse_args()
    results = run_tuning(args.config, args.quality_config)
    qualifying = [
        result
        for result in results
        if result.candidate_recall_at_10 >= 0.9 and result.resumes_covered == 100
    ]
    best = min(
        qualifying,
        key=lambda result: (result.candidate_pairs, result.runtime_seconds),
        default=None,
    )
    print(json.dumps({"best_qualifying": asdict(best) if best else None}, indent=2))


if __name__ == "__main__":
    main()
