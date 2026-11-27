from pathlib import Path

import numpy as np
import pytest

from rdkit_rag.config import Config
from rdkit_rag.linking import (
    LINK_SOURCES,
    DefinitionIndex,
    looks_like_code_name,
    short_name,
)
from rdkit_rag.models import Chunk
from rdkit_rag.retriever import Retriever
from rdkit_rag.store import save_chunks, save_embeddings


def _chunk(path: str, symbol: str | None, content: str, language: str = "C++") -> Chunk:
    return Chunk(
        id=f"{path}-{symbol}",
        file_path=path,
        language=language,
        symbol_name=symbol,
        start_line=1,
        end_line=5,
        content=content,
    )


def _doc(path: str, content: str) -> Chunk:
    return _chunk(path, "Section", content, language="Markdown")


@pytest.mark.parametrize(
    "token, expected",
    [
        ("sanitizeMol", True),
        ("MolToSmiles", True),
        ("GetSSSR", True),
        ("find_rings", True),
        ("Molecule", False),
        ("RDKit", False),
        ("ring", False),
        ("__init__", False),
    ],
)
def test_looks_like_code_name(token, expected):
    assert looks_like_code_name(token) is expected


def test_short_name():
    assert short_name("MolOps::sanitizeMol") == "sanitizeMol"
    assert short_name("SaltRemover.StripMol") == "StripMol"
    assert short_name("findSSSR") == "findSSSR"


def test_definitions_skip_docs_and_headers_and_prefer_non_test_code():
    chunks = [
        _chunk("Code/MolOps.h", "sanitizeMol", "void sanitizeMol(RWMol &mol);"),
        _chunk("Code/test/testMolOps.cpp", "sanitizeMol", "void sanitizeMol() {}"),
        _chunk("Code/MolOps.cpp", "MolOps::sanitizeMol", "void sanitizeMol(RWMol &mol) {}"),
        _doc("Docs/book.md", "Call SanitizeMol to clean up."),
    ]
    index = DefinitionIndex(chunks)

    (link,) = index.links([3])

    assert link.definition == 2  # the .cpp definition, not the header or test
    assert link.name == "SanitizeMol"  # spelled as in the doc
    assert link.source == 3


def test_links_only_follow_the_top_describing_chunks():
    defs = [_chunk(f"Code/f{n}.cpp", f"ringStep{n}", "{}") for n in range(LINK_SOURCES + 1)]
    docs = [_doc(f"Docs/d{n}.md", f"see ringStep{n}") for n in range(LINK_SOURCES + 1)]
    code_hit = _chunk("Code/other.cpp", "other", "calls ringStep3")  # not describing
    chunks = defs + docs + [code_hit]
    index = DefinitionIndex(chunks)
    n_defs = len(defs)
    ranking = [len(chunks) - 1] + [n_defs + n for n in range(LINK_SOURCES + 1)]

    linked = {link.definition for link in index.links(ranking)}

    # The first LINK_SOURCES docs are used, the last one isn't; the code hit
    # is ignored even though it ranks first.
    assert linked == set(range(LINK_SOURCES))


def test_links_ignore_plain_words_and_rank_by_mentions():
    chunks = [
        _chunk("Code/a.cpp", "Molecule", "class Molecule {}"),
        _chunk("Code/b.cpp", "findSSSR", "{}"),
        _chunk("Code/c.cpp", "fastFindRings", "{}"),
        _doc("Docs/a.md", "A Molecule uses fastFindRings or findSSSR."),
        _doc("Docs/b.md", "The SSSR comes from findSSSR."),
    ]
    links = DefinitionIndex(chunks).links([3, 4])

    assert [link.definition for link in links] == [1, 2]  # findSSSR: 2 mentions


def test_hybrid_search_adds_the_definition_a_doc_mentions(tmp_path: Path):
    config = Config(data_dir=tmp_path)
    chunks = [
        _doc("Docs/rings.md", "Rings are perceived with the findSSSR algorithm."),
        _chunk("Code/FindRings.cpp", "findSSSR", "int findSSSR(const ROMol &m) { bfs(); }"),
        _chunk("Code/Other.cpp", "unrelated", "void unrelated() {}"),
    ]
    save_chunks(chunks, config.chunks_path)
    # The doc is the closest by meaning; the definition is the farthest.
    save_embeddings(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype="float32"),
        config.embeddings_path,
    )

    class Query:
        def embed_query(self, text):
            return np.array([1.0, 0.0], dtype="float32")

    retriever = Retriever(config, embedding_service=Query())
    results = retriever.search("how are rings perceived", top_k=3, mode="hybrid")

    by_symbol = {r.chunk.symbol_name: r for r in results}
    assert by_symbol["findSSSR"].linked_from == "findSSSR in Docs/rings.md"
    assert by_symbol["Section"].linked_from is None
    # The definition is farthest by meaning, but the link lifts it above
    # the unrelated chunk.
    order = [r.chunk.symbol_name for r in results]
    assert order.index("findSSSR") < order.index("unrelated")

    # Vector and keyword modes are unchanged: no links.
    for mode in ("vector", "keyword"):
        assert all(r.linked_from is None for r in retriever.search("rings", 3, mode))
