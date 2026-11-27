"""Orchestrates the indexing pipeline: scan -> chunk -> embed -> save.

Indexing is incremental. Scanning and chunking the whole repository takes
seconds; embedding takes minutes. So every run re-chunks everything, then
reuses the stored embedding of any chunk whose embedded text is unchanged
and embeds only new or changed chunks. The result is the same as a full
rebuild, but updating after a typical merge embeds only a few dozen chunks.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunker import chunk_file
from .config import Config
from .embeddings import EmbeddingService, build_contextual_text
from .keyword import KeywordIndex
from .models import Chunk
from .scanner import scan_repository
from .store import (
    load_chunks,
    load_embeddings,
    load_metadata,
    make_metadata,
    save_chunks,
    save_embeddings,
    save_metadata,
)


class RepositoryNotFoundError(Exception):
    pass


@dataclass
class IndexResult:
    repository_name: str
    files_scanned: int
    files_indexed: int
    chunks_created: int
    chunks_embedded: int
    chunks_reused: int
    # True when no previous embeddings could be reused (first run, --force,
    # a different embedding model, or an unreadable old index).
    full_rebuild: bool
    embedding_model: str
    repository_commit: str | None
    data_dir: Path


def embedding_key(text: str) -> str:
    """Cache key for an embedded text. The text already contains the file
    path, symbol, line range and content (see build_contextual_text), so equal
    keys mean the embedding would be identical."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(repo_path: Path) -> str | None:
    """The checked-out commit, or None if repo_path isn't a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _reusable_embeddings(config: Config) -> dict[str, np.ndarray]:
    """Embeddings from the existing index, keyed by embedding_key. Empty if
    there is no usable index or it was built with a different model."""
    metadata = load_metadata(config.metadata_path)
    if metadata is None or metadata.embedding_model != config.embedding_model:
        return {}
    try:
        chunks = load_chunks(config.chunks_path)
        embeddings = load_embeddings(config.embeddings_path)
    except (OSError, ValueError):
        return {}
    if not chunks or embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        return {}
    return {
        embedding_key(build_contextual_text(chunk)): embeddings[row]
        for row, chunk in enumerate(chunks)
    }


def run_index(
    repo_path: Path,
    config: Config,
    force: bool = False,
    embedding_service: EmbeddingService | None = None,
) -> IndexResult:
    """Build the index, or update an existing one. force=True ignores any
    existing embeddings and re-embeds everything."""
    repo_path = repo_path.resolve()
    if not repo_path.is_dir():
        raise RepositoryNotFoundError(f"Not a directory: {repo_path}")

    relative_paths = scan_repository(repo_path, config.max_file_size_bytes)

    all_chunks: list[Chunk] = []
    files_indexed = 0
    for relative_path in relative_paths:
        file_chunks = chunk_file(
            repo_path, relative_path, max_chunk_lines=config.max_chunk_lines
        )
        if file_chunks:
            files_indexed += 1
            all_chunks.extend(file_chunks)

    cache = {} if force else _reusable_embeddings(config)

    texts = [build_contextual_text(chunk) for chunk in all_chunks]
    rows: list[np.ndarray | None] = [cache.get(embedding_key(t)) for t in texts]
    missing = [i for i, row in enumerate(rows) if row is None]

    if missing:
        # The model is only loaded when something actually needs embedding,
        # so an update with no changes doesn't pay the model start-up cost.
        service = embedding_service or EmbeddingService(config.embedding_model)
        new_vectors = service.embed_texts([texts[i] for i in missing])
        for i, vector in zip(missing, new_vectors):
            rows[i] = vector

    if all_chunks:
        embeddings = np.vstack(rows).astype(np.float32, copy=False)
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)

    chunks_reused = len(all_chunks) - len(missing)
    commit = git_commit(repo_path)

    # Metadata is written last: if anything before it fails, the previous
    # metadata stays in place, and a mismatch between chunks and embeddings
    # just means the next run rebuilds instead of reusing.
    config.data_dir.mkdir(parents=True, exist_ok=True)
    save_chunks(all_chunks, config.chunks_path)
    save_embeddings(embeddings, config.embeddings_path)
    KeywordIndex.build(all_chunks).save(
        config.keyword_weights_path, config.keyword_vocab_path
    )
    metadata = make_metadata(
        repo_path,
        len(all_chunks),
        config.embedding_model,
        repository_commit=commit,
        chunks_reused=chunks_reused,
    )
    save_metadata(metadata, config.metadata_path)

    return IndexResult(
        repository_name=repo_path.name,
        files_scanned=len(relative_paths),
        files_indexed=files_indexed,
        chunks_created=len(all_chunks),
        chunks_embedded=len(missing),
        chunks_reused=chunks_reused,
        full_rebuild=not cache,
        embedding_model=config.embedding_model,
        repository_commit=commit,
        data_dir=config.data_dir,
    )
