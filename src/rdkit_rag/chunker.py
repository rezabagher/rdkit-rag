"""Split source and documentation files into meaningful chunks.

Each language gets its own small, purpose-built chunker rather than one
universal fixed-size splitter:

- Python: parsed with the ``ast`` module (functions, classes, methods).
- C/C++: a lightweight heuristic (signature detection + brace counting),
  falling back to line-based chunks when parsing is uncertain.
- Markdown/RST: split by headings.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from .models import Chunk
from .scanner import CPP_EXTENSIONS, PYTHON_EXTENSIONS, language_for_extension

# A "raw chunk" is a plain dict before it's wrapped into a Chunk model. Kept
# as a dict (rather than a partial Chunk) so the per-language functions don't
# need to know about file_path/language/id generation.
RawChunk = dict


# --------------------------------------------------------------------------
# Python chunking
# --------------------------------------------------------------------------


def chunk_python(source: str, max_chunk_lines: int = 200) -> list[RawChunk]:
    """Extract module-level functions, classes' methods as chunks.

    Large functions/methods are split into line-based subchunks that keep
    the same symbol_name, per max_chunk_lines.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Don't fail the whole index over one unparsable file.
        return []

    lines = source.splitlines()
    results: list[RawChunk] = []

    def add_function(node: ast.AST, symbol_name: str) -> None:
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        content_lines = lines[start - 1 : end]

        if len(content_lines) <= max_chunk_lines:
            results.append(
                {
                    "start_line": start,
                    "end_line": end,
                    "symbol_name": symbol_name,
                    "content": "\n".join(content_lines),
                }
            )
            return

        # Function too large: split into line-based subchunks, same symbol.
        for sub_start in range(0, len(content_lines), max_chunk_lines):
            sub_end = min(sub_start + max_chunk_lines, len(content_lines))
            results.append(
                {
                    "start_line": start + sub_start,
                    "end_line": start + sub_end - 1,
                    "symbol_name": symbol_name,
                    "content": "\n".join(content_lines[sub_start:sub_end]),
                }
            )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_function(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_function(sub, f"{node.name}.{sub.name}")

    return results


# --------------------------------------------------------------------------
# C / C++ chunking (heuristic, no tree-sitter)
# --------------------------------------------------------------------------

# Matches a plausible function/method signature line, e.g.:
#   void MolOps::sanitizeMol(RWMol &mol, unsigned int &operationThatFailed) {
#   RDKIT_GRAPHMOL_EXPORT bool isAromatic(const Atom &atom) const {
_FUNC_SIGNATURE_RE = re.compile(
    r"^\s*"
    r"(?:template\s*<[^>]*>\s*)?"  # optional template<...>
    r"[\w:\<\>\*&~,\s]+?\s+"  # return type / qualifiers
    r"([A-Za-z_~][\w:]*)\s*"  # captured function/method name
    r"\([^;{}]*\)\s*"  # parameter list, no ';' or braces inside
    r"(?:const\s*)?(?:override\s*)?(?:noexcept\s*)?"
    r"\{?\s*$"
)

_CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "else",
    "do",
    "return",
    "sizeof",
}


def _find_cpp_functions(lines: list[str]) -> list[RawChunk]:
    n = len(lines)
    results: list[RawChunk] = []
    i = 0
    while i < n:
        match = _FUNC_SIGNATURE_RE.match(lines[i])
        if not match or match.group(1) in _CONTROL_KEYWORDS:
            i += 1
            continue

        symbol_name = match.group(1)

        # Look ahead a few lines for the opening brace. If we hit a ';'
        # first, this was a declaration/prototype, not a definition.
        brace_line_idx = None
        for look in range(i, min(i + 4, n)):
            if "{" in lines[look]:
                brace_line_idx = look
                break
            if lines[look].rstrip().endswith(";"):
                break

        if brace_line_idx is None:
            i += 1
            continue

        depth = 0
        end_idx = None
        for j in range(brace_line_idx, n):
            depth += lines[j].count("{")
            depth -= lines[j].count("}")
            if depth == 0:
                end_idx = j
                break

        if end_idx is None:
            # Unbalanced braces (e.g. truncated file) - stop scanning here
            # rather than guessing wrong boundaries.
            break

        results.append(
            {
                "start_line": i + 1,
                "end_line": end_idx + 1,
                "symbol_name": symbol_name,
                "content": "\n".join(lines[i : end_idx + 1]),
            }
        )
        i = end_idx + 1

    return results


def _fallback_line_chunks(
    lines: list[str], max_chunk_lines: int, overlap_lines: int
) -> list[RawChunk]:
    n = len(lines)
    if n == 0:
        return []
    results: list[RawChunk] = []
    start = 0
    while start < n:
        end = min(start + max_chunk_lines, n)
        results.append(
            {
                "start_line": start + 1,
                "end_line": end,
                "symbol_name": None,
                "content": "\n".join(lines[start:end]),
            }
        )
        if end >= n:
            break
        start = max(end - overlap_lines, start + 1)
    return results


def chunk_cpp(
    source: str, max_chunk_lines: int = 300, overlap_lines: int = 30
) -> list[RawChunk]:
    """Heuristically chunk a C/C++ file by function, falling back to
    line-based chunks (with a small overlap) when no functions are found.
    """
    lines = source.splitlines()
    try:
        chunks = _find_cpp_functions(lines)
    except Exception:
        # Never let a parsing edge case fail the whole index.
        chunks = []

    if not chunks:
        return _fallback_line_chunks(lines, max_chunk_lines, overlap_lines)

    return chunks


# --------------------------------------------------------------------------
# Documentation chunking (Markdown / RST) - split by headings
# --------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_RST_UNDERLINE_CHARS = set("=-~^\"'#*+.:_`")


def _chunk_by_heading_indices(
    lines: list[str], heading_idxs: list[tuple[int, str]]
) -> list[RawChunk]:
    n = len(lines)
    if not heading_idxs:
        if any(line.strip() for line in lines):
            return [
                {
                    "start_line": 1,
                    "end_line": n,
                    "symbol_name": None,
                    "content": "\n".join(lines),
                }
            ]
        return []

    results: list[RawChunk] = []
    for idx, (start_idx, heading) in enumerate(heading_idxs):
        end_idx = (
            heading_idxs[idx + 1][0] - 1 if idx + 1 < len(heading_idxs) else n - 1
        )
        results.append(
            {
                "start_line": start_idx + 1,
                "end_line": end_idx + 1,
                "symbol_name": heading,
                "content": "\n".join(lines[start_idx : end_idx + 1]),
            }
        )
    return results


def chunk_markdown(source: str) -> list[RawChunk]:
    lines = source.splitlines()
    headings = []
    for i, line in enumerate(lines):
        m = _MD_HEADING_RE.match(line)
        if m:
            headings.append((i, m.group(2).strip()))
    return _chunk_by_heading_indices(lines, headings)


def chunk_rst(source: str) -> list[RawChunk]:
    lines = source.splitlines()
    headings = []
    for i in range(len(lines) - 1):
        text = lines[i].strip()
        underline = lines[i + 1].strip()
        if not text or not underline:
            continue
        if (
            len(set(underline)) == 1
            and underline[0] in _RST_UNDERLINE_CHARS
            and len(underline) >= len(text)
        ):
            headings.append((i, text))
    return _chunk_by_heading_indices(lines, headings)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def chunk_file(
    repo_root: Path, relative_path: str, max_chunk_lines: int = 200
) -> list[Chunk]:
    """Read and chunk a single repository-relative file, returning Chunk
    models. Never raises - unreadable or unparsable files just yield no
    chunks so indexing the rest of the repository can continue.
    """
    full_path = repo_root / relative_path
    try:
        source = full_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    ext = full_path.suffix
    language = language_for_extension(ext)

    if ext in PYTHON_EXTENSIONS:
        raw_chunks = chunk_python(source, max_chunk_lines=max_chunk_lines)
    elif ext in CPP_EXTENSIONS:
        raw_chunks = chunk_cpp(source, max_chunk_lines=max(max_chunk_lines, 300))
    elif ext == ".md":
        raw_chunks = chunk_markdown(source)
    elif ext == ".rst":
        raw_chunks = chunk_rst(source)
    else:
        raw_chunks = []

    chunks: list[Chunk] = []
    for raw in raw_chunks:
        if not raw["content"].strip():
            continue
        chunk_id = Chunk.make_id(
            relative_path, raw["symbol_name"], raw["start_line"], raw["end_line"]
        )
        chunks.append(
            Chunk(
                id=chunk_id,
                file_path=relative_path,
                language=language,
                symbol_name=raw["symbol_name"],
                start_line=raw["start_line"],
                end_line=raw["end_line"],
                content=raw["content"],
            )
        )
    return chunks
