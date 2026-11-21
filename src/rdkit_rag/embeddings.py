"""Local embedding generation using sentence-transformers.

There is only one embedding implementation in this MVP, so this is a plain
class rather than an abstract provider interface.
"""

from __future__ import annotations

import numpy as np

from .models import Chunk


def build_contextual_text(chunk: Chunk) -> str:
    """Build the text that actually gets embedded for a chunk.

    Raw source/documentation content alone is a weak embedding target, so
    each chunk is embedded together with a small header describing where it
    came from. The *original* chunk.content is still stored separately in
    the chunk metadata (chunks.jsonl) - this function only affects what gets
    fed to the embedding model.
    """
    label = "Symbol" if chunk.language in ("Python", "C++") else "Section"
    symbol_line = f"{label}: {chunk.symbol_name}\n" if chunk.symbol_name else ""
    return (
        f"File: {chunk.file_path}\n"
        f"Language: {chunk.language}\n"
        f"{symbol_line}"
        f"Lines: {chunk.start_line}-{chunk.end_line}\n\n"
        f"Content:\n\n{chunk.content}"
    )


# BGE v1.5 English models are trained with an instruction prefix for short
# retrieval queries (passages are embedded without it). See the BAAI model
# cards for bge-*-en-v1.5.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def query_instruction_for(model_name: str) -> str:
    name = model_name.lower()
    if "bge-" in name and "-en" in name:
        return _BGE_QUERY_INSTRUCTION
    return ""


class EmbeddingService:
    """Wraps a single sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None  # lazy-loaded so `--help` etc. don't need torch

    def _load(self):
        if self._model is None:
            # Imported lazily so importing this module (e.g. for `search`
            # without embeddings, or CLI help) doesn't require torch to be
            # importable if the command doesn't actually need it.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed many texts at once (used during indexing)."""
        model = self._load()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string."""
        model = self._load()
        embedding = model.encode(
            [query_instruction_for(self.model_name) + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embedding[0], dtype=np.float32)
