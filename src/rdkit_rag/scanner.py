"""Recursively scan a local repository for supported source files."""

from __future__ import annotations

from pathlib import Path

PYTHON_EXTENSIONS = {".py"}
CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".h", ".hpp"}
DOC_EXTENSIONS = {".md", ".rst"}

SUPPORTED_EXTENSIONS = PYTHON_EXTENSIONS | CPP_EXTENSIONS | DOC_EXTENSIONS

IGNORED_DIRS = {".git", "build", "__pycache__", ".venv", "venv"}


def language_for_extension(extension: str) -> str:
    if extension in PYTHON_EXTENSIONS:
        return "Python"
    if extension in CPP_EXTENSIONS:
        return "C++"
    if extension == ".md":
        return "Markdown"
    if extension == ".rst":
        return "reStructuredText"
    return "unknown"


def _is_probably_binary(path: Path, sniff_bytes: int = 2048) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(sniff_bytes)
    except OSError:
        return True
    return b"\x00" in chunk


def scan_repository(
    repo_root: Path,
    max_file_size_bytes: int = 1_000_000,
) -> list[str]:
    """Walk repo_root and return repository-relative paths of supported
    files, skipping ignored directories, oversized files, and binaries.
    """
    repo_root = repo_root.resolve()
    results: list[str] = []

    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(repo_root)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue

        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if path.stat().st_size > max_file_size_bytes:
                continue
        except OSError:
            continue

        if _is_probably_binary(path):
            continue

        results.append(relative.as_posix())

    return results
