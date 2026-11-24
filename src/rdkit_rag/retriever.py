"""Search over the locally stored index: vector (cosine), keyword (BM25), or
both combined with reciprocal rank fusion ("hybrid", the default).

This is intentionally not optimized for millions of vectors - it's an MVP
for searching a single repository's worth of chunks in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .embeddings import EmbeddingService
from .keyword import KeywordIndex, chunks_fingerprint, reciprocal_rank_fusion
from .models import Chunk
from .store import load_chunks, load_embeddings, load_metadata

SEARCH_MODES = ("hybrid", "vector", "keyword")

# In hybrid mode each method contributes this many candidates to the fusion,
# so a chunk ranked moderately well by both methods can still surface.
HYBRID_CANDIDATES = 50


class EmptyIndexError(Exception):
    pass


class CorruptIndexError(Exception):
    pass


@dataclass
class SearchResult:
    chunk: Chunk
    # Cosine similarity (vector mode), BM25 score (keyword mode), or the
    # reciprocal-rank-fusion score (hybrid mode).
    score: float
    # 1-based rank in each underlying method, None if it didn't rank the
    # chunk among its candidates.
    vector_rank: int | None = None
    keyword_rank: int | None = None


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
        self._keyword_index: KeywordIndex | None = None

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

    @property
    def keyword_index(self) -> KeywordIndex:
        """The BM25 index for the current chunks. Indexes built before
        keyword search existed (or whose chunks changed) get it built from
        chunks.jsonl and cached next to them; no re-embedding is needed."""
        if self._keyword_index is None:
            assert self._chunks is not None
            fingerprint = chunks_fingerprint(self._chunks)
            index = KeywordIndex.load(
                self.config.keyword_weights_path, self.config.keyword_vocab_path
            )
            if index is None or index.fingerprint != fingerprint:
                index = KeywordIndex.build(self._chunks)
                try:
                    index.save(
                        self.config.keyword_weights_path, self.config.keyword_vocab_path
                    )
                except OSError:
                    pass  # read-only data dir: keep the in-memory index
            self._keyword_index = index
        return self._keyword_index

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

    def _vector_hits(self, query: str, top_k: int) -> list[tuple[int, float]]:
        assert self._embeddings is not None
        query_embedding = self.embedding_service.embed_query(query)
        if query_embedding.shape[-1] != self._embeddings.shape[1]:
            raise CorruptIndexError(
                f"Query embedding dimension ({query_embedding.shape[-1]}) does not "
                f"match the index ({self._embeddings.shape[1]}). The index was "
                "probably built with a different embedding model; rebuild it with "
                "--force."
            )
        return cosine_similarity_search(query_embedding, self._embeddings, top_k)

    def search(self, query: str, top_k: int = 8, mode: str = "hybrid") -> list[SearchResult]:
        if mode not in SEARCH_MODES:
            raise ValueError(f"Unknown search mode {mode!r}; expected one of {SEARCH_MODES}")
        self._load()
        assert self._chunks is not None
        chunks = self._chunks

        if mode == "vector":
            hits = self._vector_hits(query, top_k)
            return [
                SearchResult(chunk=chunks[i], score=s, vector_rank=r)
                for r, (i, s) in enumerate(hits, start=1)
            ]

        if mode == "keyword":
            hits = self.keyword_index.search(query, top_k)
            return [
                SearchResult(chunk=chunks[i], score=s, keyword_rank=r)
                for r, (i, s) in enumerate(hits, start=1)
            ]

        candidates = max(top_k, HYBRID_CANDIDATES)
        vector_hits = self._vector_hits(query, candidates)
        keyword_hits = self.keyword_index.search(query, candidates)
        vector_ranks = {i: r for r, (i, _) in enumerate(vector_hits, start=1)}
        keyword_ranks = {i: r for r, (i, _) in enumerate(keyword_hits, start=1)}
        fused = reciprocal_rank_fusion(
            [[i for i, _ in vector_hits], [i for i, _ in keyword_hits]]
        )
        return [
            SearchResult(
                chunk=chunks[i],
                score=score,
                vector_rank=vector_ranks.get(i),
                keyword_rank=keyword_ranks.get(i),
            )
            for i, score in fused[:top_k]
        ]
