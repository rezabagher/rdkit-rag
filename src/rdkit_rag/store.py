"""Plain local storage: chunks.jsonl + embeddings.npy + index_metadata.json.

All writes are atomic (temporary file + rename).

Kept as simple functions rather than a Store class/interface - there's only
one storage backend in this MVP (the local filesystem).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator

import numpy as np
from pydantic import BaseModel

from .models import Chunk


class IndexMetadata(BaseModel):
    repository_path: str
    repository_name: str
    num_chunks: int
    embedding_model: str
    indexed_at: str
    # Added with incremental indexing; defaults keep older metadata loadable.
    repository_commit: str | None = None
    chunks_reused: int = 0


@contextmanager
def atomic_write(path: Path, mode: str = "w") -> Iterator[IO]:
    """Write to a temporary file next to path and move it into place only
    if writing succeeded, so an interrupted (re)index never leaves a
    half-written file behind. mode is "w" (text) or "wb" (binary)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        if "b" in mode:
            with tmp.open(mode) as f:
                yield f
        else:
            with tmp.open(mode, encoding="utf-8") as f:
                yield f
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    with atomic_write(path) as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json())
            f.write("\n")


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk.model_validate_json(line))
    return chunks


def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    # Saved through a file handle so numpy doesn't append ".npy" to the
    # temporary file's name.
    with atomic_write(path, "wb") as f:
        np.save(f, embeddings)


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 0), dtype="float32")
    return np.load(path)


def save_metadata(metadata: IndexMetadata, path: Path) -> None:
    with atomic_write(path) as f:
        f.write(metadata.model_dump_json(indent=2))


def load_metadata(path: Path) -> IndexMetadata | None:
    """The index metadata, or None if it is missing or unreadable (e.g.
    truncated). Callers then fall back to defaults; indexing rebuilds."""
    try:
        return IndexMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pydantic's ValidationError is a ValueError
        return None


def make_metadata(
    repository_path: Path,
    num_chunks: int,
    embedding_model: str,
    repository_commit: str | None = None,
    chunks_reused: int = 0,
) -> IndexMetadata:
    return IndexMetadata(
        repository_path=str(repository_path.resolve()),
        repository_name=repository_path.resolve().name,
        num_chunks=num_chunks,
        embedding_model=embedding_model,
        indexed_at=datetime.now(timezone.utc).isoformat(),
        repository_commit=repository_commit,
        chunks_reused=chunks_reused,
    )


def index_exists(chunks_path: Path, embeddings_path: Path) -> bool:
    return chunks_path.exists() and embeddings_path.exists()
