"""Environment-based configuration for rdkit-rag.

Everything is configured through environment variables (optionally loaded
from a .env file). No API keys are required since indexing and answering
both run entirely locally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Load a .env file from the current working directory (or its parents) if
# present. usecwd=True matters: without it python-dotenv searches upward from
# *this module's* location, not from where the command is run. This is a
# no-op if no .env file exists, so it's safe to call unconditionally.
load_dotenv(find_dotenv(usecwd=True))


@dataclass(frozen=True)
class Config:
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:8b")
    )
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RDKIT_RAG_DATA_DIR", ".data"))
    )
    max_file_size_bytes: int = field(
        default_factory=lambda: int(
            os.getenv("RDKIT_RAG_MAX_FILE_SIZE_BYTES", str(1_000_000))
        )
    )
    max_chunk_lines: int = field(
        default_factory=lambda: int(os.getenv("RDKIT_RAG_MAX_CHUNK_LINES", "200"))
    )

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "embeddings.npy"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "index_metadata.json"

    @property
    def keyword_weights_path(self) -> Path:
        return self.data_dir / "bm25_weights.npz"

    @property
    def keyword_vocab_path(self) -> Path:
        return self.data_dir / "bm25_vocab.json"


def get_config() -> Config:
    """Return a fresh Config read from the current environment.

    Kept as a function (rather than a module-level singleton) so tests can
    monkeypatch environment variables and get a clean config each time.
    """
    return Config()
