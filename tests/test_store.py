from pathlib import Path

import numpy as np
import pytest

from rdkit_rag.models import Chunk
from rdkit_rag.store import (
    atomic_write,
    load_chunks,
    load_embeddings,
    load_metadata,
    make_metadata,
    save_chunks,
    save_embeddings,
    save_metadata,
)


def _sample_chunk(i: int) -> Chunk:
    return Chunk(
        id=f"chunk-{i}",
        file_path=f"Code/GraphMol/File{i}.cpp",
        language="C++",
        symbol_name=f"func{i}",
        start_line=1,
        end_line=10,
        content=f"void func{i}() {{}}",
    )


def test_save_and_load_chunks_roundtrip(tmp_path: Path):
    chunks = [_sample_chunk(1), _sample_chunk(2)]
    path = tmp_path / "chunks.jsonl"

    save_chunks(chunks, path)
    loaded = load_chunks(path)

    assert loaded == chunks


def test_load_chunks_missing_file_returns_empty(tmp_path: Path):
    assert load_chunks(tmp_path / "missing.jsonl") == []


def test_save_and_load_embeddings_roundtrip(tmp_path: Path):
    embeddings = np.random.rand(5, 8).astype("float32")
    path = tmp_path / "embeddings.npy"

    save_embeddings(embeddings, path)
    loaded = load_embeddings(path)

    assert np.allclose(loaded, embeddings)


def test_save_and_load_metadata_roundtrip(tmp_path: Path):
    repo = tmp_path / "rdkit"
    repo.mkdir()
    metadata = make_metadata(repo, num_chunks=42, embedding_model="test-model")
    path = tmp_path / "index_metadata.json"

    save_metadata(metadata, path)
    loaded = load_metadata(path)

    assert loaded is not None
    assert loaded.num_chunks == 42
    assert loaded.embedding_model == "test-model"
    assert loaded.repository_name == "rdkit"


def test_metadata_from_before_incremental_indexing_still_loads(tmp_path: Path):
    path = tmp_path / "index_metadata.json"
    path.write_text(
        '{"repository_path": "/x/rdkit", "repository_name": "rdkit", '
        '"num_chunks": 3, "embedding_model": "m", "indexed_at": "2026-11-21"}'
    )

    loaded = load_metadata(path)

    assert loaded.repository_commit is None
    assert loaded.chunks_reused == 0


def test_failed_write_keeps_the_previous_file(tmp_path: Path):
    path = tmp_path / "chunks.jsonl"
    save_chunks([_sample_chunk(1)], path)

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with atomic_write(path) as f:
            f.write("partial")
            raise Boom()

    assert load_chunks(path) == [_sample_chunk(1)]
    assert not list(tmp_path.glob("*.tmp"))


def test_saved_embeddings_keep_their_file_name(tmp_path: Path):
    path = tmp_path / "embeddings.npy"
    save_embeddings(np.ones((2, 3), dtype="float32"), path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["embeddings.npy"]


def test_truncated_metadata_loads_as_missing(tmp_path: Path):
    path = tmp_path / "index_metadata.json"
    path.write_text('{"repository_path": "/x"')

    assert load_metadata(path) is None
