"""Orchestrates the full indexing pipeline: scan -> chunk -> embed -> save."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunker import chunk_file
from .config import Config
from .embeddings import EmbeddingService, build_contextual_text
from .keyword import KeywordIndex
from .models import Chunk
from .scanner import scan_repository
from .store import index_exists, make_metadata, save_chunks, save_embeddings, save_metadata


class IndexAlreadyExistsError(Exception):
    pass


class RepositoryNotFoundError(Exception):
    pass


@dataclass
class IndexResult:
    repository_name: str
    files_scanned: int
    files_indexed: int
    chunks_created: int
    embedding_model: str
    data_dir: Path


def run_index(repo_path: Path, config: Config, force: bool = False) -> IndexResult:
    repo_path = repo_path.resolve()
    if not repo_path.is_dir():
        raise RepositoryNotFoundError(f"Not a directory: {repo_path}")

    if index_exists(config.chunks_path, config.embeddings_path) and not force:
        raise IndexAlreadyExistsError(
            f"An index already exists at {config.data_dir}. Use --force to rebuild it."
        )

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

    embedding_service = EmbeddingService(config.embedding_model)

    if all_chunks:
        texts = [build_contextual_text(chunk) for chunk in all_chunks]
        embeddings = embedding_service.embed_texts(texts)
    else:
        embeddings = None

    config.data_dir.mkdir(parents=True, exist_ok=True)
    save_chunks(all_chunks, config.chunks_path)
    if embeddings is not None:
        save_embeddings(embeddings, config.embeddings_path)
    else:
        import numpy as np

        save_embeddings(np.zeros((0, 0), dtype="float32"), config.embeddings_path)

    KeywordIndex.build(all_chunks).save(
        config.keyword_weights_path, config.keyword_vocab_path
    )

    metadata = make_metadata(repo_path, len(all_chunks), config.embedding_model)
    save_metadata(metadata, config.metadata_path)

    return IndexResult(
        repository_name=repo_path.name,
        files_scanned=len(relative_paths),
        files_indexed=files_indexed,
        chunks_created=len(all_chunks),
        embedding_model=config.embedding_model,
        data_dir=config.data_dir,
    )
