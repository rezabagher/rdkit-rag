"""NumPy cosine similarity search over the locally stored index.

This is intentionally not optimized for millions of vectors - it's an MVP
for searching a single repository's worth of chunks in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .embeddings import EmbeddingService
from .models import Chunk
from .store import load_chunks, load_embeddings


class EmptyIndexError(Exception):
    pass


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def cosine_similarity_search(
    query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int
) -> list[tuple[int, float]]:
    """Return (index, score) pairs for the top_k most similar rows in
    embeddings, sorted by descending similarity."""
    if embeddings.size == 0:
        return []

    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []
    query_unit = query_embedding / query_norm

    matrix_norms = np.linalg.norm(embeddings, axis=1)
    matrix_norms[matrix_norms == 0] = 1e-12  # avoid division by zero
    matrix_unit = embeddings / matrix_norms[:, None]

    scores = matrix_unit @ query_unit
    top_k = min(top_k, len(scores))
    top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [(int(i), float(scores[i])) for i in top_indices]


class Retriever:
    def __init__(self, config: Config, embedding_service: EmbeddingService | None = None):
        self.config = config
        self.embedding_service = embedding_service or EmbeddingService(
            config.embedding_model
        )
        self._chunks: list[Chunk] | None = None
        self._embeddings: np.ndarray | None = None

    def _load(self) -> None:
        if self._chunks is None:
            self._chunks = load_chunks(self.config.chunks_path)
        if self._embeddings is None:
            self._embeddings = load_embeddings(self.config.embeddings_path)
        if not self._chunks or self._embeddings.size == 0:
            raise EmptyIndexError(
                "No index found (or the index is empty). "
                "Run 'rdkit-rag index /path/to/rdkit' first."
            )

    def search(self, query: str, top_k: int = 8) -> list[SearchResult]:
        self._load()
        assert self._chunks is not None and self._embeddings is not None

        query_embedding = self.embedding_service.embed_query(query)
        hits = cosine_similarity_search(query_embedding, self._embeddings, top_k)
        return [SearchResult(chunk=self._chunks[i], score=score) for i, score in hits]
