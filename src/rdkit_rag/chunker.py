"""Split source and documentation files into meaningful chunks.

Each language gets its own small, purpose-built chunker rather than one
universal fixed-size splitter:

- Python: parsed with the ``ast`` module (functions, classes, methods).
- C/C++: a lightweight heuristic (signature detection + brace counting on
  comment/string-stripped lines), falling back to line-based chunks when
  parsing is uncertain.
- Markdown/RST: split by headings.

In every language, lines that don't belong to a detected symbol (module
docstrings, top-level code, class declarations in headers, ...) are still
indexed as "gap" chunks so no meaningful content is silently dropped.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

from .models import Chunk
from .scanner import CPP_EXTENSIONS, PYTHON_EXTENSIONS, language_for_extension

# A "raw chunk" is a plain dict before it's wrapped into a Chunk model. Kept
# as a dict (rather than a partial Chunk) so the per-language functions don't
# need to know about file_path/language/id generation.
RawChunk = dict

# Gap regions with fewer "real" lines than this (e.g. a lone closing brace,
# `}  // namespace RDKit`, or a single import) are not worth a chunk.
_MIN_GAP_CONTENT_LINES = 2


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _line_chunks(
    lines: list[str],
    start_idx: int,
    end_idx: int,
    symbol_name: str | None,
    max_chunk_lines: int,
    overlap_lines: int = 0,
) -> list[RawChunk]:
    """Split lines[start_idx..end_idx] (0-based, inclusive) into chunks of at
    most max_chunk_lines lines, each tagged with symbol_name."""
    max_chunk_lines = max(1, max_chunk_lines)
    overlap_lines = max(0, min(overlap_lines, max_chunk_lines - 1))
    results: list[RawChunk] = []
    start = start_idx
    while start <= end_idx:
        end = min(start + max_chunk_lines - 1, end_idx)
        results.append(
            {
                "start_line": start + 1,
                "end_line": end + 1,
                "symbol_name": symbol_name,
                "content": "\n".join(lines[start : end + 1]),
            }
        )
        if end >= end_idx:
            break
        start = max(end + 1 - overlap_lines, start + 1)
    return results


def _has_meaningful_content(lines: list[str], min_lines: int) -> bool:
    count = sum(1 for line in lines if any(ch.isalnum() for ch in line))
    return count >= min_lines


def _gap_chunks(
    lines: list[str],
    covered: list[bool],
    start_idx: int,
    end_idx: int,
    symbol_name: str | None,
    max_chunk_lines: int,
    overlap_lines: int = 0,
    min_content_lines: int = _MIN_GAP_CONTENT_LINES,
) -> list[RawChunk]:
    """Emit chunks for each contiguous run of uncovered lines within
    [start_idx, end_idx]. Leading/trailing blank lines are trimmed."""
    results: list[RawChunk] = []
    i = start_idx
    while i <= end_idx:
        if covered[i]:
            i += 1
            continue
        run_start = i
        while i <= end_idx and not covered[i]:
            i += 1
        run_end = i - 1
        while run_start <= run_end and not lines[run_start].strip():
            run_start += 1
        while run_end >= run_start and not lines[run_end].strip():
            run_end -= 1
        if run_start > run_end:
            continue
        if not _has_meaningful_content(lines[run_start : run_end + 1], min_content_lines):
            continue
        results.extend(
            _line_chunks(
                lines, run_start, run_end, symbol_name, max_chunk_lines, overlap_lines
            )
        )
    return results


def _sort_chunks(chunks: list[RawChunk]) -> list[RawChunk]:
    return sorted(chunks, key=lambda c: (c["start_line"], c["end_line"]))


# --------------------------------------------------------------------------
# Python chunking
# --------------------------------------------------------------------------


def _node_start(node: ast.AST) -> int:
    """1-based start line including decorators (ast's lineno points at the
    `def`/`class` line, after any decorators)."""
    start = node.lineno
    for deco in getattr(node, "decorator_list", []):
        start = min(start, deco.lineno)
    return start


def chunk_python(source: str, max_chunk_lines: int = 200) -> list[RawChunk]:
    """Extract module-level functions and classes' methods as chunks.

    Large functions/methods are split into line-based subchunks that keep
    the same symbol_name, per max_chunk_lines. Class-level code outside of
    methods (docstring, attributes) becomes chunks named after the class,
    and leftover module-level code becomes chunks without a symbol. Files
    that don't parse fall back to plain line-based chunks.
    """
    lines = source.splitlines()
    n = len(lines)
    if n == 0:
        return []

    try:
        # Old code often has invalid escape sequences ("\d" in non-raw
        # strings); don't spam SyntaxWarnings to stderr while indexing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        # e.g. Python 2 files, or generated code nested too deeply for ast. Still index them, just without symbols.
        return _line_chunks(lines, 0, n - 1, None, max_chunk_lines)

    results: list[RawChunk] = []
    covered = [False] * n

    def add_span(node: ast.AST, symbol_name: str) -> None:
        start = _node_start(node)
        end = getattr(node, "end_lineno", None) or node.lineno
        results.extend(_line_chunks(lines, start - 1, end - 1, symbol_name, max_chunk_lines))
        for idx in range(start - 1, min(end, n)):
            covered[idx] = True

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_span(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_span(sub, f"{node.name}.{sub.name}")
            # Remaining class lines (header, docstring, attributes, nested
            # classes) are attributed to the class itself. A bare
            # `class Foo:` line on its own isn't worth a chunk.
            cls_start = _node_start(node) - 1
            cls_end = (getattr(node, "end_lineno", None) or node.lineno) - 1
            results.extend(
                _gap_chunks(lines, covered, cls_start, cls_end, node.name, max_chunk_lines)
            )
            for idx in range(cls_start, cls_end + 1):
                covered[idx] = True

    results.extend(_gap_chunks(lines, covered, 0, n - 1, None, max_chunk_lines))
    return _sort_chunks(results)


# --------------------------------------------------------------------------
# C / C++ chunking (heuristic, no tree-sitter)
# --------------------------------------------------------------------------

# Matches the head of a plausible function/method definition signature
# (possibly joined from several physical lines), e.g.:
#   void MolOps::sanitizeMol(RWMol &mol, unsigned int &operationThatFailed) {
#   RDKIT_GRAPHMOL_EXPORT bool isAromatic(const Atom &atom) const {
#   ROMol *SmilesToMol(const std::string &smi,
#                      const SmilesParserParams &params) {
#   Conformer::Conformer(const Conformer &other) : d_id(other.d_id) {
# Everything after the parameter list ("tail") is validated separately by
# _valid_signature_tail: keeping several optional whitespace-led groups in
# one regex made non-matching lines backtrack catastrophically.
_FUNC_HEAD_RE = re.compile(
    r"^\s*"
    r"(?:template\s*<[^;{}]*>\s*)?"  # optional template<...>
    r"(?P<type>[A-Za-z_][\w:<>,*&\s]*?[\s*&])?"  # return type / qualifiers
    r"\s*(?P<name>[A-Za-z_~][\w:~]*)\s*"  # function/method name
    r"\((?P<params>[^;{}]*)\)"  # parameter list, no ';' or braces inside
    r"(?P<tail>.*)$"
)

_QUALIFIER_RE = re.compile(
    r"(?:const|override|final|volatile|noexcept(?:\s*\([^;{}()]*\))?)(?!\w)|&&|&"
)
_TRAILING_RETURN_RE = re.compile(r"[\w:<>,*&\s]+")

# Signatures longer than this are not worth regex-matching (and pathological
# lines, e.g. generated code, can't slow indexing down).
_MAX_SIGNATURE_CHARS = 600

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
    "new",
    "delete",
    "throw",
    "case",
    "decltype",
    "alignof",
    "static_assert",
    "defined",
}

# Words that can't be the last token of a return type (they'd mean the
# "signature" is really a statement or expression).
_BAD_TYPE_TAIL = {"return", "else", "new", "delete", "throw", "case", "goto", "co_return"}

# Macro-style blocks such as Catch2's TEST_CASE("name", "[tag]") { ... }
_MACRO_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FIRST_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
# Class/struct names declared in a file, to recognize in-class constructors.
_CLASS_NAME_RE = re.compile(
    r"\b(?:class|struct)\s+(?:[A-Z][A-Z0-9_]*\s+)?([A-Za-z_]\w*)\s*(?:final\s*)?[:{]"
)

# How many physical lines a signature may span before we give up.
_MAX_SIGNATURE_LINES = 8
# How many lines after the signature we look for the opening brace.
_MAX_BRACE_LOOKAHEAD = 6
# How many lines a `template <...>` header above a function may span.
_MAX_TEMPLATE_LINES = 4


def _is_digit_separator(line: str, i: int) -> bool:
    """True if the ' at line[i] is a C++14 digit separator (1'000'000)."""
    if i == 0 or i + 1 >= len(line) or not line[i + 1].isalnum():
        return False
    j = i - 1
    while j >= 0 and (line[j].isalnum() or line[j] in "'."):
        j -= 1
    token_start = line[j + 1 : i]
    return bool(token_start) and token_start[0].isdigit()


def _strip_cpp_noise(lines: list[str]) -> list[str]:
    """Return lines with comments removed and string/char literal contents
    blanked, so brace and paren counting isn't fooled by e.g. "{" or // }.
    The number of lines is preserved. Raw string literals aren't handled.
    """
    out: list[str] = []
    in_block_comment = False
    for line in lines:
        res: list[str] = []
        i = 0
        n = len(line)
        while i < n:
            if in_block_comment:
                end = line.find("*/", i)
                if end == -1:
                    i = n
                    break
                in_block_comment = False
                i = end + 2
                res.append(" ")
                continue
            if line.startswith("//", i):
                break
            if line.startswith("/*", i):
                in_block_comment = True
                i += 2
                continue
            c = line[i]
            if c == "'" and _is_digit_separator(line, i):
                i += 1
                continue
            if c in "\"'":
                quote = c
                res.append(quote)
                i += 1
                while i < n and line[i] != quote:
                    i += 2 if line[i] == "\\" else 1
                res.append(quote)
                i += 1
                continue
            res.append(c)
            i += 1
        out.append("".join(res))
    return out


def _valid_signature_tail(tail: str) -> tuple[bool, bool]:
    """Check what follows a parameter list: qualifiers, a trailing return
    type, a constructor initializer list, and/or an opening brace (possibly
    with a one-line body). Returns (is_valid, has_initializer_list)."""
    if "{" in tail:
        pre = tail[: tail.index("{")]
    else:
        pre = tail
    if ";" in pre:
        return False, False
    pre = pre.strip()
    while pre:
        m = _QUALIFIER_RE.match(pre)
        if m:
            pre = pre[m.end() :].lstrip()
            continue
        if pre.startswith("->"):
            return bool(_TRAILING_RETURN_RE.fullmatch(pre[2:])), False
        if pre.startswith(":") and not pre.startswith("::"):
            return True, True  # initializer list
        return False, False
    return True, False


def _match_signature(
    clean: list[str], original: list[str], i: int, class_names: set[str]
) -> tuple[str, int] | None:
    """If a function definition signature starts at line i, return
    (symbol_name, index of the signature's last line)."""
    first = clean[i]
    if "(" not in first or first.lstrip().startswith("#"):
        return None

    # Join continuation lines until the parentheses balance.
    sig = first
    sig_end = i
    while sig.count("(") > sig.count(")") and sig_end - i < _MAX_SIGNATURE_LINES:
        if sig_end + 1 >= len(clean):
            return None
        sig_end += 1
        sig += " " + clean[sig_end].strip()
    if sig.count("(") > sig.count(")") or len(sig) > _MAX_SIGNATURE_CHARS:
        return None

    m = _FUNC_HEAD_RE.match(sig)
    if not m:
        return None
    tail_ok, has_init_list = _valid_signature_tail(m.group("tail"))
    if not tail_ok:
        return None

    name = m.group("name")
    type_part = (m.group("type") or "").strip()
    if name in _CONTROL_KEYWORDS:
        return None
    if type_part and type_part.split()[-1].strip("*&") in _BAD_TYPE_TAIL:
        return None

    if not type_part:
        if "::" in name or name.startswith("~"):
            pass  # constructor/destructor: Foo::Foo(...) {, ~Foo() {
        elif name in class_names or has_init_list:
            pass  # in-class constructor: Foo(int x) : d_x(x) {
        elif (
            _MACRO_NAME_RE.match(name)
            and not first[:1].isspace()
            and sig.rstrip().endswith("{")
        ):
            # Top-level macro block, e.g. TEST_CASE("SMILES parsing") {
            label = _FIRST_STRING_RE.search(" ".join(original[i : sig_end + 1]))
            if label:
                name = f'{name}("{label.group(1)}")'
        else:
            return None

    return name, sig_end


def _template_header_start(clean: list[str], start_idx: int) -> int:
    """If a (possibly multi-line) `template <...>` header sits directly above
    start_idx, return the index of its first line; otherwise start_idx."""
    for k in range(1, _MAX_TEMPLATE_LINES + 1):
        cand = start_idx - k
        if cand < 0:
            break
        text = clean[cand]
        if any(ch in text for ch in ";{}"):
            break
        if text.lstrip().startswith("template"):
            header = " ".join(clean[cand:start_idx])
            if header.count("<") == header.count(">"):
                return cand
            break
    return start_idx


def _find_cpp_functions(lines: list[str], clean: list[str]) -> list[tuple[int, int, str]]:
    """Return (start_idx, end_idx, symbol_name) spans, 0-based inclusive."""
    n = len(lines)
    class_names = set(_CLASS_NAME_RE.findall("\n".join(clean)))
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        sig = _match_signature(clean, lines, i, class_names)
        if sig is None:
            i += 1
            continue
        symbol_name, sig_end = sig

        # Find the opening brace: on the signature itself or within a few
        # lines after it (e.g. Allman style, or a multi-line initializer
        # list). A ';' first means this was a declaration, not a definition.
        brace_line_idx = None
        if "{" in clean[sig_end]:
            brace_line_idx = sig_end
        else:
            for look in range(sig_end + 1, min(sig_end + 1 + _MAX_BRACE_LOOKAHEAD, n)):
                text = clean[look]
                brace_pos = text.find("{")
                semi_pos = text.find(";")
                if semi_pos != -1 and (brace_pos == -1 or semi_pos < brace_pos):
                    break
                if brace_pos != -1:
                    brace_line_idx = look
                    break

        if brace_line_idx is None:
            i += 1
            continue

        depth = 0
        end_idx = None
        for j in range(i, n):
            depth += clean[j].count("{") - clean[j].count("}")
            if j >= brace_line_idx and depth <= 0:
                end_idx = j
                break

        if end_idx is None:
            # Unbalanced braces (e.g. macros or #ifdef'd code confused the
            # counter). Skip this candidate but keep scanning the file.
            i += 1
            continue

        spans.append((_template_header_start(clean, i), end_idx, symbol_name))
        i = end_idx + 1

    return spans


def _fallback_line_chunks(
    lines: list[str], max_chunk_lines: int, overlap_lines: int
) -> list[RawChunk]:
    if not lines:
        return []
    return _line_chunks(lines, 0, len(lines) - 1, None, max_chunk_lines, overlap_lines)


def chunk_cpp(
    source: str, max_chunk_lines: int = 300, overlap_lines: int = 30
) -> list[RawChunk]:
    """Heuristically chunk a C/C++ file by function. Code outside detected
    functions (class declarations, doc comments, globals) is kept as
    line-based chunks with a small overlap; if no functions are found at all
    the whole file is chunked that way.
    """
    lines = source.splitlines()
    if not lines:
        return []
    try:
        clean = _strip_cpp_noise(lines)
        spans = _find_cpp_functions(lines, clean)
    except Exception:
        # Never let a parsing edge case fail the whole index.
        spans = []

    if not spans:
        return _fallback_line_chunks(lines, max_chunk_lines, overlap_lines)

    results: list[RawChunk] = []
    covered = [False] * len(lines)
    for start_idx, end_idx, symbol_name in spans:
        results.extend(_line_chunks(lines, start_idx, end_idx, symbol_name, max_chunk_lines))
        for idx in range(start_idx, end_idx + 1):
            covered[idx] = True

    results.extend(
        _gap_chunks(lines, covered, 0, len(lines) - 1, None, max_chunk_lines, overlap_lines)
    )
    return _sort_chunks(results)


# --------------------------------------------------------------------------
# Documentation chunking (Markdown / RST) - split by headings
# --------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_MD_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_RST_UNDERLINE_CHARS = set("=-~^\"'#*+.:_`")


def _chunk_by_heading_indices(
    lines: list[str], heading_idxs: list[tuple[int, str]], max_chunk_lines: int
) -> list[RawChunk]:
    n = len(lines)
    results: list[RawChunk] = []

    # Text before the first heading (or the whole file if there are none).
    first = heading_idxs[0][0] if heading_idxs else n
    if first > 0 and any(line.strip() for line in lines[:first]):
        results.extend(_line_chunks(lines, 0, first - 1, None, max_chunk_lines))

    for idx, (start_idx, heading) in enumerate(heading_idxs):
        end_idx = (
            heading_idxs[idx + 1][0] - 1 if idx + 1 < len(heading_idxs) else n - 1
        )
        results.extend(_line_chunks(lines, start_idx, end_idx, heading, max_chunk_lines))
    return results


def chunk_markdown(source: str, max_chunk_lines: int = 200) -> list[RawChunk]:
    lines = source.splitlines()
    headings = []
    fence: str | None = None
    for i, line in enumerate(lines):
        fm = _MD_FENCE_RE.match(line)
        if fm:
            marker = fm.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is not None:
            # `# comment` lines inside a fenced code block aren't headings.
            continue
        m = _MD_HEADING_RE.match(line)
        if m and m.group(2):
            headings.append((i, m.group(2).strip()))
    return _chunk_by_heading_indices(lines, headings, max_chunk_lines)


def chunk_rst(source: str, max_chunk_lines: int = 200) -> list[RawChunk]:
    lines = source.splitlines()
    headings = []
    for i in range(len(lines) - 1):
        raw = lines[i]
        text = raw.strip()
        underline = lines[i + 1].strip()
        if not text or not underline:
            continue
        has_overline = i > 0 and lines[i - 1].strip() == underline
        if raw[:1].isspace() and not has_overline:
            continue
        # A line made only of punctuation is an overline/transition, not a
        # title (otherwise "=====" over "=====" would be a heading).
        if set(text) <= _RST_UNDERLINE_CHARS:
            continue
        if (
            len(set(underline)) == 1
            and underline[0] in _RST_UNDERLINE_CHARS
            and len(underline) >= len(text)
        ):
            # Over-and-underlined title: include the overline.
            start = i - 1 if has_overline else i
            headings.append((start, text))
    return _chunk_by_heading_indices(lines, headings, max_chunk_lines)


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

    try:
        if ext in PYTHON_EXTENSIONS:
            raw_chunks = chunk_python(source, max_chunk_lines=max_chunk_lines)
        elif ext in CPP_EXTENSIONS:
            raw_chunks = chunk_cpp(source, max_chunk_lines=max_chunk_lines)
        elif ext == ".md":
            raw_chunks = chunk_markdown(source, max_chunk_lines=max_chunk_lines)
        elif ext == ".rst":
            raw_chunks = chunk_rst(source, max_chunk_lines=max_chunk_lines)
        else:
            raw_chunks = []
    except Exception:
        # e.g. RecursionError from ast on pathological input.
        return []

    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    for raw in raw_chunks:
        if not raw["content"].strip():
            continue
        chunk_id = Chunk.make_id(
            relative_path, raw["symbol_name"], raw["start_line"], raw["end_line"]
        )
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
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
