"""Data models for rdkit-rag."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel


class Chunk(BaseModel):
    """A single retrievable unit of code or documentation."""

    id: str
    file_path: str
    language: str
    symbol_name: str | None = None
    start_line: int
    end_line: int
    content: str

    @staticmethod
    def make_id(
        file_path: str, symbol_name: str | None, start_line: int, end_line: int
    ) -> str:
        """Deterministic id derived from location, so re-indexing the same
        repository state produces the same ids."""
        key = f"{file_path}:{symbol_name or ''}:{start_line}:{end_line}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
