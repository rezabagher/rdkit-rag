from pathlib import Path

import numpy as np
import pytest

from rdkit_rag.config import Config
from rdkit_rag.keyword import (
    KeywordIndex,
    _stem,
    is_test_path,
    reciprocal_rank_fusion,
    tokenize,
)
from rdkit_rag.models import Chunk
from rdkit_rag.retriever import Retriever
from rdkit_rag.store import save_chunks, save_embeddings


def _chunk(i: int, path: str, symbol: str | None, content: str) -> Chunk:
    return Chunk(
        id=f"c{i}",
        file_path=path,
        language="C++",
        symbol_name=symbol,
        start_line=1,
        end_line=5,
        content=content,
    )


CHUNKS = [
    _chunk(0, "Docs/Book/Intro.md", "Overview", "RDKit reads molecules and writes them out again."),
    _chunk(
        1,
        "Code/GraphMol/SmilesParse/SmilesWrite.cpp",
        "MolToSmiles",
        "std::string MolToSmiles(const ROMol &mol) { return canonicalize(mol); }",
    ),
    _chunk(
        2,
        "Code/GraphMol/Fingerprints/MorganGenerator.cpp",
        "getMorganGenerator",
        "FingerprintGenerator *getMorganGenerator(unsigned int radius) { ... }",
    ),
    _chunk(3, "Code/GraphMol/MolOps.cpp", "sanitize_mol", "void sanitize_mol(RWMol &mol) { kekulize(mol); }"),
]


class FakeEmbeddingService:
    def __init__(self, vector=None):
        self.vector = vector
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        if self.vector is None:
            raise AssertionError("embedding model must not be used in keyword mode")
        return self.vector


def test_tokenize_splits_identifiers_but_keeps_them_whole():
    tokens = tokenize("MolToSmiles(sanitize_mol)")
    assert _stem("moltosmiles") in tokens  # whole identifier
    assert {"mol", _stem("smiles"), "sanitize_mol", _stem("sanitize")} <= set(tokens)
    assert "to" not in tokens  # stopword


def test_tokenize_stems_so_words_meet_identifiers():
    assert "fingerprint" in tokenize("Where are Morgan fingerprints implemented?")
    assert "fingerprint" in tokenize("FingerprintGenerator")
    assert "where" not in tokenize("Where")


@pytest.mark.parametrize(
    "words",
    [
        ("kekulization", "kekulize", "kekulized"),
        ("parser", "parse", "parsing", "parsed"),
        ("embedding", "embed", "embedded"),
        ("sanitization", "sanitize"),
        ("rings", "ring"),
    ],
)
def test_stemmer_conflates_related_forms(words):
    assert len({_stem(w) for w in words}) == 1


def test_stemmer_leaves_identifiers_with_digits_or_underscores():
    assert _stem("sanitize_mol") == "sanitize_mol"
    assert _stem("utf8encoding") == "utf8encoding"


@pytest.mark.parametrize(
    "path, expected",
    [
        ("Code/GraphMol/catch_graphmol.cpp", True),
        ("Code/GraphMol/molopstest.cpp", True),
        ("Code/GraphMol/SmilesParse/cxsmiles_test.cpp", True),
        ("Code/GraphMol/Fingerprints/testFingerprintGenerators.cpp", True),
        ("rdkit/Chem/UnitTestChem.py", True),
        ("Code/Bench/smiles.cpp", True),
        ("Code/GraphMol/MolOps.cpp", False),
        ("Code/GraphMol/ContestUtils.cpp", False),
        ("Code/GraphMol/EnumerateStereoisomers/Flippers.cpp", False),
    ],
)
def test_is_test_path(path, expected):
    assert is_test_path(path) is expected


def test_test_files_rank_below_implementation_for_same_words():
    chunks = [
        _chunk(0, "Code/GraphMol/catch_kekulize.cpp", 'TEST_CASE("kekulize ring")', "kekulize ring"),
        _chunk(1, "Code/GraphMol/Kekulize.cpp", "kekulize", "kekulize ring"),
    ]
    hits = KeywordIndex.build(chunks).search("kekulize ring", top_k=2)
    assert [i for i, _ in hits] == [1, 0]


def test_bm25_finds_exact_identifier_first():
    index = KeywordIndex.build(CHUNKS)
    hits = index.search("MolToSmiles", top_k=4)
    assert hits[0][0] == 1


def test_bm25_uses_paths_and_symbols():
    index = KeywordIndex.build(CHUNKS)
    hits = index.search("morgan fingerprints", top_k=4)
    assert hits[0][0] == 2


def test_bm25_returns_nothing_without_shared_words():
    index = KeywordIndex.build(CHUNKS)
    assert index.search("zeolite crystallography", top_k=4) == []


def test_keyword_index_save_load_roundtrip(tmp_path: Path):
    index = KeywordIndex.build(CHUNKS)
    index.save(tmp_path / "w.npz", tmp_path / "v.json")
    loaded = KeywordIndex.load(tmp_path / "w.npz", tmp_path / "v.json")
    assert loaded is not None
    assert loaded.fingerprint == index.fingerprint
    assert np.allclose(loaded.scores("sanitize mol"), index.scores("sanitize mol"))


def test_reciprocal_rank_fusion_rewards_agreement():
    # Chunk 7 is second in both lists; chunks 1 and 2 each top only one list.
    fused = reciprocal_rank_fusion([[1, 7, 3], [2, 7, 4]])
    assert fused[0][0] == 7
    assert {i for i, _ in fused} == {1, 2, 3, 4, 7}


def _make_index(tmp_path: Path) -> Config:
    config = Config(data_dir=tmp_path)
    save_chunks(CHUNKS, config.chunks_path)
    # Vectors deliberately point at the docs chunk (0) for every query, so
    # only keyword search can surface the exact identifier.
    embeddings = np.eye(4, dtype="float32")
    save_embeddings(embeddings, config.embeddings_path)
    return config


def test_hybrid_surfaces_identifier_that_vector_search_misses(tmp_path: Path):
    config = _make_index(tmp_path)
    fake = FakeEmbeddingService(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))
    retriever = Retriever(config, embedding_service=fake)

    vector_top = retriever.search("MolToSmiles", top_k=1, mode="vector")[0]
    assert vector_top.chunk.id == "c0"

    hybrid = retriever.search("MolToSmiles", top_k=2, mode="hybrid")
    ids = [r.chunk.id for r in hybrid]
    assert "c1" in ids
    c1 = next(r for r in hybrid if r.chunk.id == "c1")
    assert c1.keyword_rank == 1 and c1.vector_rank is not None


def test_keyword_mode_does_not_load_embedding_model(tmp_path: Path):
    config = _make_index(tmp_path)
    fake = FakeEmbeddingService(vector=None)
    results = Retriever(config, embedding_service=fake).search(
        "sanitize_mol", top_k=2, mode="keyword"
    )
    assert results[0].chunk.id == "c3"
    assert fake.calls == 0


def test_keyword_index_is_built_and_cached_for_old_indexes(tmp_path: Path):
    config = _make_index(tmp_path)
    assert not config.keyword_weights_path.exists()
    Retriever(config, embedding_service=FakeEmbeddingService()).search(
        "MolToSmiles", top_k=1, mode="keyword"
    )
    assert config.keyword_weights_path.exists()


def test_stale_keyword_index_is_rebuilt(tmp_path: Path):
    config = _make_index(tmp_path)
    KeywordIndex.build(CHUNKS[:2]).save(config.keyword_weights_path, config.keyword_vocab_path)
    results = Retriever(config, embedding_service=FakeEmbeddingService()).search(
        "sanitize_mol", top_k=1, mode="keyword"
    )
    assert results[0].chunk.id == "c3"


def test_unknown_mode_is_rejected(tmp_path: Path):
    config = _make_index(tmp_path)
    with pytest.raises(ValueError):
        Retriever(config, embedding_service=FakeEmbeddingService()).search("x", mode="fuzzy")
