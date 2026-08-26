"""Plain local storage: chunks.jsonl + embeddings.npy + index_metadata.json.

Kept as simple functions rather than a Store class/interface - there's only
one storage backend in this MVP (the local filesystem).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .models import Chunk


class IndexMetadata(BaseModel):
    repository_path: str
    repository_name: str
    num_chunks: int
    embedding_model: str
    indexed_at: str


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 0), dtype="float32")
    return np.load(path)


def save_metadata(metadata: IndexMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")


def load_metadata(path: Path) -> IndexMetadata | None:
    if not path.exists():
        return None
    return IndexMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def make_metadata(
    repository_path: Path, num_chunks: int, embedding_model: str
) -> IndexMetadata:
    return IndexMetadata(
        repository_path=str(repository_path.resolve()),
        repository_name=repository_path.resolve().name,
        num_chunks=num_chunks,
        embedding_model=embedding_model,
        indexed_at=datetime.now(timezone.utc).isoformat(),
    )


def index_exists(chunks_path: Path, embeddings_path: Path) -> bool:
    return chunks_path.exists() and embeddings_path.exists()
