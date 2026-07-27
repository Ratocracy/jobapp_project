from jobapps.relevance_review import build_blinded_packet, select_diverse_resumes


def test_select_diverse_resumes_prefers_unique_roles() -> None:
    metadata = {
        "r1": {"resume_role": "engineer"},
        "r2": {"resume_role": "engineer"},
        "r3": {"resume_role": "nurse"},
    }

    selected = select_diverse_resumes(metadata, set(metadata), count=2, seed=5110)

    assert {metadata[resume_id]["resume_role"] for resume_id in selected} == {
        "engineer",
        "nurse",
    }


def test_build_blinded_packet_pools_and_hides_provenance() -> None:
    tfidf = [
        {"resume_id": "r1", "job_link": "j1", "rank": 1, "similarity_score": 0.8},
        {"resume_id": "r1", "job_link": "j2", "rank": 2, "similarity_score": 0.7},
    ]
    transformer = [
        {"resume_id": "r1", "job_link": "j1", "rank": 2, "similarity_score": 0.6},
        {"resume_id": "r1", "job_link": "j3", "rank": 1, "similarity_score": 0.9},
    ]
    ngram = [
        {"resume_id": "r1", "job_link": "j2", "rank": 1, "similarity_score": 0.75},
        {"resume_id": "r1", "job_link": "j4", "rank": 2, "similarity_score": 0.65},
    ]
    reviewer, answer = build_blinded_packet(
        tfidf,
        ngram,
        transformer,
        {"r1": {"resume_role": "engineer", "resume_excerpt": "python"}},
        {
            "j1": {"job_title": "one"},
            "j2": {"job_title": "two"},
            "j3": {"job_title": "three"},
            "j4": {"job_title": "four"},
        },
        resume_count=1,
        seed=5110,
    )

    assert len(reviewer) == 4
    assert len(answer) == 4
    assert all("tfidf_rank" not in row for row in reviewer)
    shared = next(row for row in answer if row["job_link"] == "j1")
    assert shared["in_tfidf"] is True
    assert shared["in_transformer"] is True
    ngram_only = next(row for row in answer if row["job_link"] == "j4")
    assert ngram_only["in_ngram"] is True
    assert ngram_only["in_tfidf"] is False
