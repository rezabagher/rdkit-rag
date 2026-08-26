from pathlib import Path

import numpy as np

from rdkit_rag.config import Config
from rdkit_rag.models import Chunk
from rdkit_rag.retriever import Retriever, cosine_similarity_search
from rdkit_rag.store import save_chunks, save_embeddings


class FakeEmbeddingService:
    """Avoids downloading a real model - returns a fixed vector per query
    that is deliberately closest to a chosen target embedding."""

    def __init__(self, query_vector: np.ndarray):
        self.query_vector = query_vector

    def embed_query(self, text: str) -> np.ndarray:
        return self.query_vector


def test_cosine_similarity_search_orders_by_similarity():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype="float32",
    )
    query = np.array([1.0, 0.0], dtype="float32")

    results = cosine_similarity_search(query, embeddings, top_k=2)

    assert results[0][0] == 0  # most similar to [1, 0]
    assert len(results) == 2


def test_retriever_returns_most_relevant_chunk(tmp_path: Path):
    config = Config(data_dir=tmp_path)

    chunks = [
        Chunk(
            id="a",
            file_path="Code/GraphMol/MolOps.cpp",
            language="C++",
            symbol_name="sanitizeMol",
            start_line=10,
            end_line=50,
            content="void sanitizeMol(RWMol &mol) { ... }",
        ),
        Chunk(
            id="b",
            file_path="Code/GraphMol/FileParsers.cpp",
            language="C++",
            symbol_name="parseSDF",
            start_line=1,
            end_line=20,
            content="Mol *parseSDF(std::string path) { ... }",
        ),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # matches chunk "a"
            [0.0, 1.0, 0.0],  # matches chunk "b"
        ],
        dtype="float32",
    )

    save_chunks(chunks, config.chunks_path)
    save_embeddings(embeddings, config.embeddings_path)

    fake_service = FakeEmbeddingService(np.array([1.0, 0.0, 0.0], dtype="float32"))
    retriever = Retriever(config, embedding_service=fake_service)

    results = retriever.search("Where is molecule sanitization implemented?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.symbol_name == "sanitizeMol"
    assert results[0].score > 0.99
