"""Batched, chunk-aware transformer embeddings for sampled documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from pyspark.sql import DataFrame, SparkSession

from jobapps.documents import require_document_contract


@dataclass(frozen=True)
class EmbeddedDocument:
    document_id: str
    document_type: str
    source_id: str
    embedding_model: str
    text_contract_version: str
    chunk_count: int
    embedding: list[float]


def load_sentence_transformer(model_name: str, device: str) -> Any:
    """Load the optional model dependency only when semantic encoding is run."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Transformer benchmarking requires sentence-transformers. "
            "Run `conda env update -f environment.yml --prune` first."
        ) from exc
    return SentenceTransformer(model_name, device=device)


def chunk_text(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into deterministic overlapping tokenizer-aware chunks."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be in [0, max_tokens)")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    step = max_tokens - overlap_tokens
    chunks = [
        tokenizer.decode(
            token_ids[start : start + max_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        for start in range(0, len(token_ids), step)
    ]
    return [chunk for chunk in chunks if chunk]


def embed_document_rows(
    rows: Iterable[Any],
    model: Any,
    model_name: str,
    batch_size: int,
    max_tokens: int,
    overlap_tokens: int,
    text_contract_version: str = "combined_text_v1",
) -> list[EmbeddedDocument]:
    """Encode document rows and mean-pool their normalized chunk embeddings."""

    materialized = list(rows)
    all_chunks: list[str] = []
    chunk_ranges: list[tuple[int, int]] = []
    for row in materialized:
        chunks = chunk_text(
            row["combined_text"],
            model.tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        if not chunks:
            raise ValueError(f"Document {row['document_id']} produced no chunks")
        start = len(all_chunks)
        all_chunks.extend(chunks)
        chunk_ranges.append((start, len(all_chunks)))

    encoded = np.asarray(
        model.encode(
            all_chunks,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    if encoded.ndim != 2 or encoded.shape[0] != len(all_chunks):
        raise ValueError("Transformer returned an unexpected embedding shape")

    output: list[EmbeddedDocument] = []
    for row, (start, end) in zip(materialized, chunk_ranges, strict=True):
        pooled = encoded[start:end].mean(axis=0)
        norm = float(np.linalg.norm(pooled))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"Document {row['document_id']} has an invalid embedding")
        pooled = pooled / norm
        output.append(
            EmbeddedDocument(
                document_id=row["document_id"],
                document_type=row["document_type"],
                source_id=row["source_id"],
                embedding_model=model_name,
                text_contract_version=text_contract_version,
                chunk_count=end - start,
                embedding=pooled.astype(np.float32).tolist(),
            )
        )
    return output


def embed_documents(
    spark: SparkSession,
    documents: DataFrame,
    model: Any,
    model_name: str,
    batch_size: int,
    max_tokens: int,
    overlap_tokens: int,
) -> DataFrame:
    """Collect a controlled sample, encode it locally, and return a Spark table."""

    require_document_contract(documents)
    rows = documents.select(
        "document_id", "document_type", "source_id", "combined_text"
    ).orderBy("document_id").collect()
    embedded = embed_document_rows(
        rows,
        model=model,
        model_name=model_name,
        batch_size=batch_size,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    return spark.createDataFrame(
        [
            (
                item.document_id,
                item.document_type,
                item.source_id,
                item.embedding_model,
                item.text_contract_version,
                item.chunk_count,
                item.embedding,
            )
            for item in embedded
        ],
        "document_id string, document_type string, source_id string, "
        "embedding_model string, text_contract_version string, "
        "chunk_count int, embedding array<float>",
    )
