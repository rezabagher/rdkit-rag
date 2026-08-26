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
