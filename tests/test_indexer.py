import hashlib
from pathlib import Path

import numpy as np
import pytest

from rdkit_rag import indexer
from rdkit_rag.config import Config
from rdkit_rag.indexer import run_index
from rdkit_rag.keyword import KeywordIndex
from rdkit_rag.store import (
    load_chunks,
    load_embeddings,
    load_metadata,
    save_embeddings,
)

PY_SOURCE = '''\
def sanitize(mol):
    """Clean up a molecule."""
    return mol


def kekulize(mol):
    """Assign alternating single and double bonds."""
    return mol
'''

MD_SOURCE = """\
# Getting started

Install the package.

# Usage

Call sanitize on every molecule.
"""


class FakeEmbeddingService:
    """Deterministic vectors derived from the text, so a reused embedding and
    a freshly computed one are equal. Records every text it is asked to embed."""

    def __init__(self):
        self.embedded: list[str] = []

    def embed_texts(self, texts):
        self.embedded.extend(texts)
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append(np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32))
        return np.vstack(rows)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "Code").mkdir(parents=True)
    (root / "Code" / "molops.py").write_text(PY_SOURCE)
    (root / "Docs").mkdir()
    (root / "Docs" / "guide.md").write_text(MD_SOURCE)
    return root


def _config(tmp_path: Path, model: str = "fake-model", name: str = "data") -> Config:
    return Config(data_dir=tmp_path / name, embedding_model=model)


def _index(repo: Path, config: Config, force: bool = False):
    service = FakeEmbeddingService()
    result = run_index(repo, config, force=force, embedding_service=service)
    return result, service


def _stored(config: Config):
    return load_chunks(config.chunks_path), load_embeddings(config.embeddings_path)


def test_first_run_embeds_every_chunk(repo, tmp_path):
    config = _config(tmp_path)
    result, service = _index(repo, config)

    chunks, embeddings = _stored(config)
    assert result.full_rebuild
    assert result.chunks_embedded == len(chunks) == len(service.embedded) > 0
    assert result.chunks_reused == 0
    assert embeddings.shape[0] == len(chunks)


def test_unchanged_repo_embeds_nothing_and_never_loads_the_model(
    repo, tmp_path, monkeypatch
):
    config = _config(tmp_path)
    _index(repo, config)
    _, before = _stored(config)

    def no_model(*args, **kwargs):
        raise AssertionError("the embedding model should not be loaded")

    monkeypatch.setattr(indexer, "EmbeddingService", no_model)
    result = run_index(repo, config)

    _, after = _stored(config)
    assert not result.full_rebuild
    assert result.chunks_embedded == 0
    assert result.chunks_reused == result.chunks_created
    assert np.array_equal(before, after)


def test_edit_only_reembeds_that_file_and_matches_full_rebuild(repo, tmp_path):
    config = _config(tmp_path)
    _index(repo, config)

    (repo / "Code" / "molops.py").write_text(
        PY_SOURCE.replace("return mol\n\n\ndef kekulize", "return mol.copy()\n\n\ndef kekulize")
    )
    result, service = _index(repo, config)

    assert 0 < result.chunks_embedded < result.chunks_created
    assert all("File: Code/molops.py" in text for text in service.embedded)

    # An update must give exactly what a from-scratch build would.
    fresh = _config(tmp_path, name="fresh")
    _index(repo, fresh, force=True)
    chunks, embeddings = _stored(config)
    fresh_chunks, fresh_embeddings = _stored(fresh)
    assert chunks == fresh_chunks
    assert np.array_equal(embeddings, fresh_embeddings)
    keyword = KeywordIndex.load(config.keyword_weights_path, config.keyword_vocab_path)
    fresh_keyword = KeywordIndex.load(fresh.keyword_weights_path, fresh.keyword_vocab_path)
    assert keyword.fingerprint == fresh_keyword.fingerprint


def test_added_and_deleted_files(repo, tmp_path):
    config = _config(tmp_path)
    _index(repo, config)

    (repo / "Docs" / "guide.md").unlink()
    (repo / "Code" / "rings.py").write_text("def find_rings(mol):\n    return []\n")
    result, service = _index(repo, config)

    chunks, embeddings = _stored(config)
    paths = {chunk.file_path for chunk in chunks}
    assert paths == {"Code/molops.py", "Code/rings.py"}
    assert embeddings.shape[0] == len(chunks)
    assert all("File: Code/rings.py" in text for text in service.embedded)
    assert result.chunks_reused > 0


def test_force_reembeds_everything(repo, tmp_path):
    config = _config(tmp_path)
    _index(repo, config)
    result, service = _index(repo, config, force=True)

    assert result.full_rebuild
    assert result.chunks_embedded == result.chunks_created == len(service.embedded)


def test_changed_embedding_model_reembeds_everything(repo, tmp_path):
    _index(repo, _config(tmp_path, model="model-a"))
    config_b = _config(tmp_path, model="model-b")
    result, _ = _index(repo, config_b)

    assert result.full_rebuild
    assert result.chunks_reused == 0
    assert load_metadata(config_b.metadata_path).embedding_model == "model-b"


def test_inconsistent_old_index_is_rebuilt_not_an_error(repo, tmp_path):
    config = _config(tmp_path)
    _index(repo, config)
    save_embeddings(np.zeros((1, 16), dtype=np.float32), config.embeddings_path)

    result, _ = _index(repo, config)

    chunks, embeddings = _stored(config)
    assert result.full_rebuild
    assert embeddings.shape[0] == len(chunks)


def test_metadata_records_reuse_and_leaves_no_temp_files(repo, tmp_path):
    config = _config(tmp_path)
    _index(repo, config)
    result, _ = _index(repo, config)

    metadata = load_metadata(config.metadata_path)
    assert metadata.chunks_reused == result.chunks_reused
    assert metadata.repository_commit is None  # tmp repo is not a git checkout
    assert not list(config.data_dir.glob("*.tmp"))
