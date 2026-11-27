import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from rdkit_rag.cli import app
from rdkit_rag.config import Config
from rdkit_rag.evaluation import (
    Question,
    QuestionSet,
    QuestionSetError,
    Target,
    evaluate,
    first_relevant_rank,
    default_questions_path,
    load_questions,
    missing_targets,
    same_commit,
    score,
)
from rdkit_rag.indexer import run_index
from rdkit_rag.models import Chunk
from rdkit_rag.retriever import SearchResult



def _chunk(path: str, symbol: str | None, i: int = 0) -> Chunk:
    return Chunk(
        id=f"{path}-{symbol}-{i}",
        file_path=path,
        language="C++",
        symbol_name=symbol,
        start_line=1,
        end_line=2,
        content="x",
    )


@pytest.mark.parametrize(
    "symbol,chunk_symbol,expected",
    [
        (None, "anything", True),
        (None, None, True),
        ("addHs", "addHs", True),
        ("calcExplicitValence", "Atom::calcExplicitValence", True),
        ("StripMol", "SaltRemover.StripMol", True),
        ("addHs", "addHsHelper", False),
        ("addHs", None, False),
    ],
)
def test_target_matching(symbol, chunk_symbol, expected):
    target = Target(file="Code/GraphMol/AddHs.cpp", symbol=symbol)
    assert target.matches(_chunk("Code/GraphMol/AddHs.cpp", chunk_symbol)) is expected


def test_target_requires_the_same_file():
    target = Target(file="Code/GraphMol/AddHs.cpp", symbol="addHs")
    assert not target.matches(_chunk("Code/GraphMol/AddHs.h", "addHs"))


def test_first_relevant_rank_uses_any_target():
    chunks = [_chunk("a.cpp", "f"), _chunk("b.cpp", "g"), _chunk("c.cpp", "h")]
    targets = (Target("c.cpp"), Target("b.cpp", "g"))
    assert first_relevant_rank(chunks, targets) == 2
    assert first_relevant_rank(chunks, (Target("z.cpp"),)) is None


def test_score_recall_and_mrr():
    scores = score([1, 3, None], ks=(1, 5))
    assert scores.recall == {1: pytest.approx(1 / 3), 5: pytest.approx(2 / 3)}
    assert scores.mrr == pytest.approx((1 + 1 / 3) / 3)
    assert scores.count == 3


class FakeSearcher:
    """Returns a fixed ranking per query and records the requested depth."""

    def __init__(self, rankings: dict[str, list[Chunk]]):
        self.rankings = rankings
        self.calls: list[tuple[str, int, str]] = []

    def search(self, query, top_k=8, mode="hybrid"):
        self.calls.append((query, top_k, mode))
        return [SearchResult(chunk=c, score=1.0) for c in self.rankings[query][:top_k]]


def test_evaluate_reports_overall_and_per_category():
    hit = _chunk("hit.cpp", "f")
    miss = _chunk("other.cpp", "g")
    question_set = QuestionSet(
        questions=[
            Question("a", "qa", "concept", (Target("hit.cpp"),)),
            Question("b", "qb", "concept", (Target("hit.cpp"),)),
            Question("c", "qc", "identifier", (Target("hit.cpp", "f"),)),
        ]
    )
    searcher = FakeSearcher({"qa": [hit], "qb": [miss, miss], "qc": [miss, hit]})

    report = evaluate(searcher, question_set, mode="keyword", ks=(5, 1))

    assert report.ks == (1, 5)
    assert report.ranks == {"a": 1, "b": None, "c": 2}
    assert report.overall.mrr == pytest.approx((1 + 0.5) / 3)
    assert report.by_category["concept"].recall[5] == pytest.approx(0.5)
    assert report.by_category["identifier"].recall[1] == 0.0
    assert {(top_k, mode) for _, top_k, mode in searcher.calls} == {(5, "keyword")}


def test_missing_targets_flags_locations_not_in_the_index():
    question_set = QuestionSet(
        questions=[
            Question("ok", "q", "c", (Target("a.cpp", "f"),)),
            Question("gone", "q", "c", (Target("a.cpp", "removed"), Target("b.cpp"))),
        ]
    )
    missing = missing_targets([_chunk("a.cpp", "f")], question_set)
    assert missing == [("gone", Target("a.cpp", "removed")), ("gone", Target("b.cpp"))]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "q.toml"
    path.write_text(text)
    return path


def test_load_questions(tmp_path):
    path = _write(
        tmp_path,
        'rdkit_commit = "abc"\n'
        '[[question]]\nid = "x"\nquery = "Where?"\ncategory = "concept"\n'
        'relevant = [{ file = "a.cpp", symbol = "f" }, { file = "b.py" }]\n',
    )
    question_set = load_questions(path)
    assert question_set.commit == "abc"
    (question,) = question_set.questions
    assert question.relevant == (Target("a.cpp", "f"), Target("b.py", None))


@pytest.mark.parametrize(
    "text,message",
    [
        ("", "No .*question.* entries"),
        ('[[question]]\nid = "x"\nquery = "q"\n', "missing 'relevant'"),
        ('[question]\nid = "x"\n', r"\[\[question\]\] tables"),
        ('question = ["x"]\n', "must be a table"),
        ('[[question]]\nid = "x"\nquery = "q"\nrelevant = ["a.cpp"]\n', "must look like"),
        ('[[question]]\nid = "x"\nquery = "q"\nrelevant = [{ file = 3 }]\n', "'file' must be"),
        ('[[question]]\nid = 7\nquery = "q"\nrelevant = [{ file = "a" }]\n', "'id' must be"),
        (
            'rdkit_commit = 123\n[[question]]\nid = "x"\nquery = "q"\n'
            'relevant = [{ file = "a" }]\n',
            "rdkit_commit must be",
        ),
        ('[[question]]\nid = "x"\nquery = "q"\nrelevant = []\n', "no relevant"),
        (
            '[[question]]\nid = "x"\nquery = "q"\nrelevant = [{ file = "a" }]\n' * 2,
            "Duplicate",
        ),
        ("not = = toml", "Invalid TOML"),
    ],
)
def test_load_questions_rejects_bad_sets(tmp_path, text, message):
    with pytest.raises(QuestionSetError, match=message):
        load_questions(_write(tmp_path, text))


def test_same_commit_allows_short_hashes():
    full = "ce08c344e86dbb07c86526cca2517436b1e41b02"
    assert same_commit(full, "ce08c344e")
    assert same_commit("CE08C344E", full)
    assert not same_commit(full, "ce08c344f")


def test_shipped_question_set_is_well_formed():
    question_set = load_questions(default_questions_path())
    assert len(question_set.questions) >= 30
    assert {q.category for q in question_set.questions} == {"concept", "identifier"}
    assert question_set.commit and len(question_set.commit) == 40
    for question in question_set.questions:
        assert question.query.strip()
        assert all(t.file and not t.file.startswith("/") for t in question.relevant)


class FakeEmbeddingService:
    def embed_texts(self, texts):
        return np.vstack(
            [
                np.frombuffer(hashlib.sha256(t.encode()).digest()[:8], dtype=np.uint8)
                for t in texts
            ]
        ).astype(np.float32)


def test_eval_command_keyword_mode_end_to_end(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "molops.py").write_text(
        "def sanitize_molecule(mol):\n    return mol\n\n\n"
        "def kekulize_rings(mol):\n    return mol\n"
    )
    data_dir = tmp_path / "data"
    run_index(
        repo,
        Config(data_dir=data_dir, embedding_model="fake"),
        embedding_service=FakeEmbeddingService(),
    )
    questions = _write(
        tmp_path,
        '[[question]]\nid = "kek"\nquery = "kekulize_rings"\ncategory = "identifier"\n'
        'relevant = [{ file = "molops.py", symbol = "kekulize_rings" }]\n'
        '[[question]]\nid = "nope"\nquery = "sanitize"\ncategory = "identifier"\n'
        'relevant = [{ file = "missing.py" }]\n',
    )
    out = tmp_path / "results.json"
    monkeypatch.setenv("RDKIT_RAG_DATA_DIR", str(data_dir))

    result = CliRunner().invoke(
        app,
        ["eval", str(questions), "--mode", "keyword", "--k", "1", "--details",
         "--output", str(out)],
    )

    assert result.exit_code == 0, result.output
    assert "no indexed chunk matches missing.py" in result.output
    payload = json.loads(out.read_text())
    (keyword,) = payload["results"]
    assert keyword["mode"] == "keyword"
    assert keyword["ranks"] == {"kek": 1, "nope": None}
    assert keyword["overall"]["recall@1"] == 0.5


def _tiny_index(tmp_path: Path, monkeypatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "molops.py").write_text("def kekulize_rings(mol):\n    return mol\n")
    data_dir = tmp_path / "data"
    run_index(
        repo,
        Config(data_dir=data_dir, embedding_model="fake"),
        embedding_service=FakeEmbeddingService(),
    )
    monkeypatch.setenv("RDKIT_RAG_DATA_DIR", str(data_dir))
    return data_dir


def test_eval_uses_built_in_questions_from_any_directory(tmp_path, monkeypatch):
    _tiny_index(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)  # no eval/ directory here

    result = CliRunner().invoke(app, ["eval", "--mode", "keyword"])

    assert result.exit_code == 0, result.output
    assert "45 questions" in result.output


def test_eval_creates_the_output_directory(tmp_path, monkeypatch):
    _tiny_index(tmp_path, monkeypatch)
    questions = _write(
        tmp_path,
        '[[question]]\nid = "k"\nquery = "kekulize_rings"\n'
        'relevant = [{ file = "molops.py" }]\n',
    )
    out = tmp_path / "results" / "run1.json"

    result = CliRunner().invoke(
        app, ["eval", str(questions), "--mode", "keyword", "--output", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["results"][0]["ranks"] == {"k": 1}


def test_eval_short_commit_in_question_set_is_not_a_mismatch(tmp_path, monkeypatch):
    data_dir = _tiny_index(tmp_path, monkeypatch)
    meta_path = data_dir / "index_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["repository_commit"] = "ce08c344e86dbb07c86526cca2517436b1e41b02"
    meta_path.write_text(json.dumps(meta))
    questions = _write(
        tmp_path,
        'rdkit_commit = "ce08c344e"\n[[question]]\nid = "k"\nquery = "kekulize_rings"\n'
        'relevant = [{ file = "molops.py" }]\n',
    )

    result = CliRunner().invoke(app, ["eval", str(questions), "--mode", "keyword"])

    assert result.exit_code == 0, result.output
    assert "Note:" not in result.output


def test_eval_survives_truncated_metadata(tmp_path, monkeypatch):
    data_dir = _tiny_index(tmp_path, monkeypatch)
    (data_dir / "index_metadata.json").write_text('{"repository_pa')
    questions = _write(
        tmp_path,
        '[[question]]\nid = "k"\nquery = "kekulize_rings"\n'
        'relevant = [{ file = "molops.py" }]\n',
    )

    result = CliRunner().invoke(app, ["eval", str(questions), "--mode", "keyword"])

    assert result.exit_code == 0, result.output
