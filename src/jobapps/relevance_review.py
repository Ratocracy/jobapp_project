"""Deterministic pooling and blinding for human relevance review."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any, Iterable


def _stable_key(seed: int, *values: str) -> str:
    payload = "|".join([str(seed), *values]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_diverse_resumes(
    resume_metadata: dict[str, dict[str, str]],
    available_resume_ids: set[str],
    count: int,
    seed: int,
) -> list[str]:
    """Select at most one deterministic resume per role before repeating roles."""

    if count < 1:
        raise ValueError("count must be positive")
    eligible = [
        resume_id
        for resume_id in available_resume_ids
        if resume_id in resume_metadata
    ]
    by_role: dict[str, list[str]] = defaultdict(list)
    for resume_id in eligible:
        role = resume_metadata[resume_id].get("resume_role", "").strip() or "unknown"
        by_role[role].append(resume_id)

    first_per_role = [
        min(ids, key=lambda resume_id: _stable_key(seed, role, resume_id))
        for role, ids in by_role.items()
    ]
    selected = sorted(
        first_per_role,
        key=lambda resume_id: _stable_key(
            seed,
            resume_metadata[resume_id].get("resume_role", ""),
            resume_id,
        ),
    )
    if len(selected) < count:
        remaining = sorted(
            set(eligible) - set(selected),
            key=lambda resume_id: _stable_key(seed, resume_id),
        )
        selected.extend(remaining[: count - len(selected)])
    return selected[:count]


def build_blinded_packet(
    tfidf_rows: Iterable[Any],
    transformer_rows: Iterable[Any],
    resume_metadata: dict[str, dict[str, str]],
    job_metadata: dict[str, dict[str, str]],
    resume_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pool two top-K systems and return reviewer rows plus a hidden answer key."""

    pooled: dict[tuple[str, str], dict[str, Any]] = {}
    available_resume_ids: set[str] = set()
    for system, rows in (("tfidf", tfidf_rows), ("transformer", transformer_rows)):
        for row in rows:
            resume_id = row["resume_id"]
            job_link = row["job_link"]
            available_resume_ids.add(resume_id)
            entry = pooled.setdefault((resume_id, job_link), {})
            entry[f"{system}_rank"] = int(row["rank"])
            entry[f"{system}_score"] = float(row["similarity_score"])

    selected_ids = select_diverse_resumes(
        resume_metadata,
        available_resume_ids,
        resume_count,
        seed,
    )
    query_ids = {resume_id: f"Q{index:03d}" for index, resume_id in enumerate(selected_ids, 1)}

    reviewer_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    for resume_id in selected_ids:
        candidates = [
            (job_link, values)
            for (candidate_resume_id, job_link), values in pooled.items()
            if candidate_resume_id == resume_id
        ]
        candidates.sort(key=lambda item: _stable_key(seed, resume_id, item[0]))
        for position, (job_link, provenance) in enumerate(candidates, 1):
            query_id = query_ids[resume_id]
            review_id = f"{query_id}-C{position:02d}"
            resume = resume_metadata[resume_id]
            job = job_metadata.get(job_link, {})
            reviewer_rows.append(
                {
                    "review_id": review_id,
                    "query_id": query_id,
                    "resume_role": resume.get("resume_role", ""),
                    "resume_excerpt": resume.get("resume_excerpt", ""),
                    "job_title": job.get("job_title", ""),
                    "company": job.get("company", ""),
                    "job_location": job.get("job_location", ""),
                    "job_level": job.get("job_level", ""),
                    "job_type": job.get("job_type", ""),
                    "job_summary_excerpt": job.get("job_summary_excerpt", ""),
                    "role_relevance_1_to_5": "",
                    "skills_alignment_1_to_5": "",
                    "seniority_alignment_1_to_5": "",
                    "overall_relevance_1_to_5": "",
                    "cannot_judge": "",
                    "reviewer_notes": "",
                }
            )
            answer_rows.append(
                {
                    "review_id": review_id,
                    "query_id": query_id,
                    "resume_id": resume_id,
                    "job_link": job_link,
                    "in_tfidf": "tfidf_rank" in provenance,
                    "tfidf_rank": provenance.get("tfidf_rank", ""),
                    "tfidf_score": provenance.get("tfidf_score", ""),
                    "in_transformer": "transformer_rank" in provenance,
                    "transformer_rank": provenance.get("transformer_rank", ""),
                    "transformer_score": provenance.get("transformer_score", ""),
                }
            )
    return reviewer_rows, answer_rows
