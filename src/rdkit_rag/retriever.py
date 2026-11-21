"""NumPy cosine similarity search over the locally stored index.

This is intentionally not optimized for millions of vectors - it's an MVP
for searching a single repository's worth of chunks in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .embeddings import EmbeddingService
from .models import Chunk
from .store import load_chunks, load_embeddings, load_metadata


class EmptyIndexError(Exception):
    pass


class CorruptIndexError(Exception):
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
    if embeddings.size == 0 or top_k <= 0:
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
        self._embedding_service = embedding_service
        self._chunks: list[Chunk] | None = None
        self._embeddings: np.ndarray | None = None

    @property
    def embedding_service(self) -> EmbeddingService:
        # Queries must be embedded with the same model the index was built
        # with, otherwise scores are meaningless (or the dimensions don't
        # even match). Prefer the model recorded in the index metadata over
        # the current EMBEDDING_MODEL setting.
        if self._embedding_service is None:
            metadata = load_metadata(self.config.metadata_path)
            model_name = metadata.embedding_model if metadata else self.config.embedding_model
            self._embedding_service = EmbeddingService(model_name)
        return self._embedding_service

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
        if self._embeddings.ndim != 2 or self._embeddings.shape[0] != len(self._chunks):
            raise CorruptIndexError(
                f"Index is inconsistent: {len(self._chunks)} chunks but embeddings "
                f"have shape {self._embeddings.shape}. Rebuild it with "
                "'rdkit-rag index /path/to/rdkit --force'."
            )

    def search(self, query: str, top_k: int = 8) -> list[SearchResult]:
        self._load()
        assert self._chunks is not None and self._embeddings is not None

        query_embedding = self.embedding_service.embed_query(query)
        if query_embedding.shape[-1] != self._embeddings.shape[1]:
            raise CorruptIndexError(
                f"Query embedding dimension ({query_embedding.shape[-1]}) does not "
                f"match the index ({self._embeddings.shape[1]}). The index was "
                "probably built with a different embedding model; rebuild it with "
                "--force."
            )
        hits = cosine_similarity_search(query_embedding, self._embeddings, top_k)
        return [SearchResult(chunk=self._chunks[i], score=score) for i, score in hits]
