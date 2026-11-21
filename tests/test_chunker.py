from rdkit_rag.chunker import chunk_cpp, chunk_markdown, chunk_python


PY_SOURCE = '''\
def top_level_function(x):
    return x + 1


class Mol:
    def __init__(self, name):
        self.name = name

    def sanitize(self):
        return True
'''


def test_python_chunker_extracts_module_function():
    chunks = chunk_python(PY_SOURCE)
    names = [c["symbol_name"] for c in chunks]
    assert "top_level_function" in names


def test_python_chunker_extracts_class_methods():
    chunks = chunk_python(PY_SOURCE)
    names = [c["symbol_name"] for c in chunks]
    assert "Mol.__init__" in names
    assert "Mol.sanitize" in names
    # The class itself should not become one giant chunk.
    assert "Mol" not in names


def test_python_chunker_records_correct_line_numbers():
    chunks = chunk_python(PY_SOURCE)
    top_level = next(c for c in chunks if c["symbol_name"] == "top_level_function")
    assert top_level["start_line"] == 1
    assert top_level["end_line"] == 2


def test_python_chunker_splits_large_functions():
    lines = ["def big():"] + [f"    x = {i}" for i in range(500)]
    source = "\n".join(lines) + "\n"
    chunks = chunk_python(source, max_chunk_lines=100)
    big_chunks = [c for c in chunks if c["symbol_name"] == "big"]
    assert len(big_chunks) > 1
    for c in big_chunks:
        assert c["symbol_name"] == "big"


CPP_SOURCE = """\
#include <string>

void sanitizeMol(RWMol &mol) {
    int depth = 0;
    if (depth == 0) {
        depth = 1;
    }
}

bool isAromatic(const Atom &atom) const {
    return true;
}
"""


def test_cpp_chunker_detects_simple_function():
    chunks = chunk_cpp(CPP_SOURCE)
    names = [c["symbol_name"] for c in chunks]
    assert "sanitizeMol" in names
    assert "isAromatic" in names


def test_cpp_chunker_handles_nested_braces():
    chunks = chunk_cpp(CPP_SOURCE)
    sanitize = next(c for c in chunks if c["symbol_name"] == "sanitizeMol")
    # The nested `if` block's braces must be included, ending at the
    # function's own closing brace, not the inner one.
    assert sanitize["content"].strip().endswith("}")
    assert sanitize["content"].count("{") == sanitize["content"].count("}")


def test_cpp_chunker_falls_back_when_no_functions_found():
    source = "\n".join(f"int x{i} = {i};" for i in range(50)) + "\n"
    chunks = chunk_cpp(source, max_chunk_lines=20, overlap_lines=5)
    assert len(chunks) > 1
    assert all(c["symbol_name"] is None for c in chunks)


MD_SOURCE = """\
# Molecule Sanitization

Some intro text.

## Steps

1. Kekulize
2. Set aromaticity

## Caveats

Watch out for edge cases.
"""


def test_markdown_chunker_splits_by_headings():
    chunks = chunk_markdown(MD_SOURCE)
    headings = [c["symbol_name"] for c in chunks]
    assert headings == ["Molecule Sanitization", "Steps", "Caveats"]
    steps_chunk = next(c for c in chunks if c["symbol_name"] == "Steps")
    assert "Kekulize" in steps_chunk["content"]


def test_python_chunker_keeps_module_level_code_and_class_body():
    source = '''\
"""Module docstring about sanitization."""
import os

DEFAULT_FLAGS = 7


@decorator
def decorated():
    return 1


class Mol:
    """A molecule."""

    kind = "mol"

    def sanitize(self):
        return True
'''
    chunks = chunk_python(source)
    by_name = {c["symbol_name"]: c for c in chunks}
    assert "decorated" in by_name
    assert by_name["decorated"]["content"].startswith("@decorator")
    assert "A molecule." in by_name["Mol"]["content"]
    assert "Mol.sanitize" in by_name
    module_chunks = [c for c in chunks if c["symbol_name"] is None]
    assert any("Module docstring" in c["content"] for c in module_chunks)


def test_python_chunker_falls_back_on_syntax_error():
    source = 'print "python 2"\nx = 1\n'
    chunks = chunk_python(source)
    assert len(chunks) == 1
    assert chunks[0]["symbol_name"] is None


CPP_RDKIT_STYLE = """\
namespace RDKit {
// a comment with a stray brace }
ROMol *SmilesToMol(const std::string &smi,
                   const SmilesParserParams &params) {
  std::string s = "{not a brace";
  return nullptr;
}

Conformer::Conformer(const Conformer &other)
    : d_id(other.d_id) {
  d_positions = other.d_positions;
}

Conformer::~Conformer() {}

void declaredOnly(int x);
}  // namespace RDKit
"""


def test_cpp_chunker_handles_pointer_returns_multiline_and_ctors():
    chunks = chunk_cpp(CPP_RDKIT_STYLE)
    by_name = {c["symbol_name"]: c for c in chunks}
    assert "SmilesToMol" in by_name
    smi = by_name["SmilesToMol"]
    assert smi["start_line"] == 3 and smi["end_line"] == 7
    assert "Conformer::Conformer" in by_name
    assert by_name["Conformer::Conformer"]["end_line"] == 12
    assert "Conformer::~Conformer" in by_name
    assert "declaredOnly" not in by_name


def test_cpp_chunker_keeps_header_declarations():
    source = """\
//! Docs for the class
class RDKIT_GRAPHMOL_EXPORT Atom {
 public:
  //! returns the atomic number
  int getAtomicNum() const { return d_atomicNum; }
  void setAtomicNum(int n);
 private:
  int d_atomicNum;
};
"""
    chunks = chunk_cpp(source)
    names = [c["symbol_name"] for c in chunks]
    assert "getAtomicNum" in names
    joined = "\n".join(c["content"] for c in chunks)
    assert "Docs for the class" in joined
    assert "setAtomicNum" in joined


def test_cpp_chunker_detects_catch2_test_cases():
    source = 'TEST_CASE("SMILES parsing", "[smiles]") {\n  REQUIRE(1);\n}\n'
    chunks = chunk_cpp(source)
    assert chunks[0]["symbol_name"] == 'TEST_CASE("SMILES parsing")'


def test_cpp_chunker_splits_large_functions():
    source = "void big() {\n" + "".join(f"  x{i}();\n" for i in range(50)) + "}\n"
    chunks = chunk_cpp(source, max_chunk_lines=20)
    assert len(chunks) == 3
    assert all(c["symbol_name"] == "big" for c in chunks)
    assert chunks[-1]["end_line"] == 52


def test_markdown_ignores_headings_in_code_fences_and_keeps_preamble():
    source = "Intro text.\n\n# Title\n\n```python\n# not a heading\nx = 1\n```\n\n## Next\nbody\n"
    chunks = chunk_markdown(source)
    names = [c["symbol_name"] for c in chunks]
    assert names == [None, "Title", "Next"]
    assert "# not a heading" in chunks[1]["content"]


def test_rst_chunker_ignores_punctuation_only_lines():
    from rdkit_rag.chunker import chunk_rst

    source = "=====\nTitle\n=====\n\ntext\n\n-----\n-----\n\nSub\n---\nmore\n"
    chunks = chunk_rst(source)
    names = [c["symbol_name"] for c in chunks]
    assert names == ["Title", "Sub"]
    assert chunks[0]["start_line"] == 1


def test_cpp_signature_regex_is_not_catastrophic():
    import time

    source = "int f(x)" + " " * 2000 + "y\n" + "int g(" + " " * 2000 + "a b c\n"
    t = time.perf_counter()
    chunk_cpp(source)
    assert time.perf_counter() - t < 1.0


def test_cpp_digit_separators_do_not_break_braces():
    source = "void foo(int x) {\n  if (x > 1'000) {\n    x = 0;\n  }\n  bar(x);\n}\n"
    chunks = chunk_cpp(source)
    assert chunks[0]["symbol_name"] == "foo"
    assert chunks[0]["end_line"] == 6


def test_cpp_multiline_template_and_in_class_ctor():
    source = """\
template <typename T,
          typename U>
T add(T a, U b) {
  return a + b;
}

class A {
 public:
  A(int x)
      : d_x(x) {}
  ~A() {}
  int d_x;
};
"""
    by_name = {c["symbol_name"]: c for c in chunk_cpp(source)}
    assert by_name["add"]["start_line"] == 1
    assert by_name["A"]["start_line"] == 9
    assert "~A" in by_name


def test_markdown_heading_keeps_trailing_hash_without_space():
    chunks = chunk_markdown("# Using C#\ntext\n## Closed ##\n")
    assert [c["symbol_name"] for c in chunks] == ["Using C#", "Closed"]


def test_rst_indented_overlined_title():
    from rdkit_rag.chunker import chunk_rst

    chunks = chunk_rst("=========\n  Title  \n=========\n\ntext\n")
    assert chunks[0]["symbol_name"] == "Title"
