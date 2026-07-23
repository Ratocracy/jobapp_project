import pytest

from jobapps.pipelines.retrieval_tuning import evaluate_candidate_rows


def test_evaluate_candidate_rows_applies_threshold_and_ranks_top_k() -> None:
    candidates = [
        ("resume-1", "job-a", 0.2),
        ("resume-1", "job-b", 0.4),
        ("resume-1", "job-c", 1.3),
        ("resume-2", "job-d", 0.3),
    ]
    exact = {
        "resume-1": {"job-a", "job-c"},
        "resume-2": {"job-d", "job-e"},
    }

    pair_count, coverage, candidate_recall, final_recall = evaluate_candidate_rows(
        candidates, exact, distance_threshold=1.2, top_k=1
    )

    assert pair_count == 3
    assert coverage == 2
    assert candidate_recall == pytest.approx(0.5)
    assert final_recall == pytest.approx(0.5)
