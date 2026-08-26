from pathlib import Path

from rdkit_rag.scanner import scan_repository


def _write(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_finds_python_files(tmp_path: Path):
    _write(tmp_path / "src" / "mod.py")
    results = scan_repository(tmp_path)
    assert "src/mod.py" in results


def test_finds_cpp_files(tmp_path: Path):
    _write(tmp_path / "Code" / "GraphMol" / "MolOps.cpp", "void f() {}\n")
    _write(tmp_path / "Code" / "GraphMol" / "MolOps.h", "void f();\n")
    results = scan_repository(tmp_path)
    assert "Code/GraphMol/MolOps.cpp" in results
    assert "Code/GraphMol/MolOps.h" in results


def test_ignores_build_directories(tmp_path: Path):
    _write(tmp_path / "build" / "generated.py")
    _write(tmp_path / ".git" / "hooks" / "pre-commit.py")
    _write(tmp_path / "__pycache__" / "cached.py")
    results = scan_repository(tmp_path)
    assert results == []


def test_ignores_unsupported_files(tmp_path: Path):
    _write(tmp_path / "image.png", "not really an image")
    _write(tmp_path / "notes.txt", "plain text")
    results = scan_repository(tmp_path)
    assert results == []


def test_ignores_files_over_size_limit(tmp_path: Path):
    _write(tmp_path / "big.py", "x = 1\n" * 100)
    results = scan_repository(tmp_path, max_file_size_bytes=10)
    assert results == []


def test_returns_repository_relative_paths(tmp_path: Path):
    _write(tmp_path / "Docs" / "sanitization.md", "# Title\n")
    results = scan_repository(tmp_path)
    assert results == ["Docs/sanitization.md"]
