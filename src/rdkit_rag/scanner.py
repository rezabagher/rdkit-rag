"""Recursively scan a local repository for supported source files."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PYTHON_EXTENSIONS = {".py"}
CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".h", ".hpp"}
DOC_EXTENSIONS = {".md", ".rst"}

SUPPORTED_EXTENSIONS = PYTHON_EXTENSIONS | CPP_EXTENSIONS | DOC_EXTENSIONS

IGNORED_DIRS = {
    ".git",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    ".data",  # rdkit-rag's own index directory
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
}


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


def _git_listed_files(repo_root: Path) -> list[Path] | None:
    """Files git considers part of the working tree (tracked + untracked but
    not ignored), or None if repo_root isn't inside a git work tree or git
    isn't installed. This skips e.g. third-party sources that RDKit's CMake
    build downloads into External/ (all .gitignore'd)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True,
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    names = proc.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    files: list[Path] = []
    for name in names:
        if not name:
            continue
        path = repo_root / name
        if path.is_dir():
            # A submodule shows up as a single gitlink entry; index its
            # contents by walking it.
            files.extend(_walk_files(path))
        else:
            files.append(path)
    return files


def _walk_files(repo_root: Path) -> list[Path]:
    # os.walk lets us prune ignored directories *before* descending into
    # them; rglob("*") would walk an entire build/ or .git/ tree first.
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        current = Path(dirpath)
        files.extend(current / filename for filename in filenames)
    return files


def scan_repository(
    repo_root: Path,
    max_file_size_bytes: int = 1_000_000,
    respect_gitignore: bool = True,
) -> list[str]:
    """Walk repo_root and return sorted repository-relative paths of
    supported files, skipping ignored directories, git-ignored files (when
    repo_root is a git checkout), oversized files, and binaries.
    """
    repo_root = repo_root.resolve()
    candidates = _git_listed_files(repo_root) if respect_gitignore else None
    if not candidates or not any(p.suffix in SUPPORTED_EXTENSIONS for p in candidates):
        # Not a git checkout, or git reports nothing useful - e.g. repo_root
        # is a plain folder that an enclosing repository ignores.
        candidates = _walk_files(repo_root)

    results: list[str] = []
    for path in candidates:
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(repo_root)
        if any(part in IGNORED_DIRS for part in relative.parts[:-1]):
            continue
        try:
            if not path.is_file() or path.stat().st_size > max_file_size_bytes:
                continue
        except OSError:
            continue
        if _is_probably_binary(path):
            continue
        results.append(relative.as_posix())

    results.sort()
    return results
