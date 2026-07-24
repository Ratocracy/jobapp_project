import numpy as np
import pytest

from jobapps.transformer_features import chunk_text, embed_document_rows


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is True
        return " ".join(token_ids)


class FakeModel:
    tokenizer = FakeTokenizer()

    def encode(
        self,
        chunks,
        batch_size,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        assert batch_size == 2
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        return np.asarray(
            [[1.0, 0.0] if "python" in chunk else [0.0, 1.0] for chunk in chunks],
            dtype=np.float32,
        )


def test_chunk_text_uses_deterministic_overlap() -> None:
    chunks = chunk_text(
        "one two three four five six",
        FakeTokenizer(),
        max_tokens=4,
        overlap_tokens=2,
    )

    assert chunks == ["one two three four", "three four five six", "five six"]


def test_embed_document_rows_pools_and_normalizes_chunks() -> None:
    rows = [
        {
            "document_id": "doc-1",
            "document_type": "resume",
            "source_id": "resume-1",
            "combined_text": "python spark leadership management",
        }
    ]

    result = embed_document_rows(
        rows,
        model=FakeModel(),
        model_name="fake-model",
        batch_size=2,
        max_tokens=2,
        overlap_tokens=0,
    )[0]

    assert result.chunk_count == 2
    assert result.embedding == [pytest.approx(2**-0.5), pytest.approx(2**-0.5)]
