"""Retrieval evaluation: run a fixed question set through search and report
Recall@k and MRR, so changes to chunking or ranking can be compared against
a baseline number.

A question set is a TOML file (see eval/rdkit_questions.toml). Each question
lists the chunks that answer it, as a file plus an optional symbol name.

Metrics, over the first max(ks) results of each question:
- Recall@k: fraction of questions with at least one relevant chunk in the
  top k. Most questions have a single relevant location, so this equals the
  usual recall; with several, any one of them counts.
- MRR: mean of 1/rank of the first relevant chunk (0 when none is found).
"""

from __future__ import annotations

import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .models import Chunk

DEFAULT_KS = (1, 5, 10)


class QuestionSetError(Exception):
    pass


@dataclass(frozen=True)
class Target:
    file: str
    symbol: str | None = None

    def matches(self, chunk: Chunk) -> bool:
        if chunk.file_path != self.file:
            return False
        if self.symbol is None:
            return True
        name = chunk.symbol_name or ""
        return (
            name == self.symbol
            or name.endswith("::" + self.symbol)
            or name.endswith("." + self.symbol)
        )


@dataclass(frozen=True)
class Question:
    id: str
    query: str
    category: str
    relevant: tuple[Target, ...]


@dataclass
class QuestionSet:
    questions: list[Question]
    # The repository commit the relevant locations were checked against.
    commit: str | None = None


def load_questions(path: Path) -> QuestionSet:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise QuestionSetError(f"Cannot read question set {path}: {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise QuestionSetError(f"Invalid TOML in {path}: {e}") from e

    questions: list[Question] = []
    seen: set[str] = set()
    for n, raw in enumerate(data.get("question", []), start=1):
        try:
            qid = str(raw["id"])
            targets = tuple(
                Target(file=t["file"], symbol=t.get("symbol")) for t in raw["relevant"]
            )
            question = Question(
                id=qid,
                query=str(raw["query"]),
                category=str(raw.get("category", "uncategorized")),
                relevant=targets,
            )
        except (KeyError, TypeError) as e:
            raise QuestionSetError(f"Question #{n} in {path} is missing {e}") from e
        if not targets:
            raise QuestionSetError(f"Question {qid!r} has no relevant locations")
        if qid in seen:
            raise QuestionSetError(f"Duplicate question id {qid!r}")
        seen.add(qid)
        questions.append(question)

    if not questions:
        raise QuestionSetError(f"No [[question]] entries in {path}")
    return QuestionSet(questions=questions, commit=data.get("rdkit_commit"))


def first_relevant_rank(chunks: list[Chunk], targets: tuple[Target, ...]) -> int | None:
    """1-based rank of the first chunk matching any target, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if any(target.matches(chunk) for target in targets):
            return rank
    return None


def missing_targets(
    chunks: list[Chunk], question_set: QuestionSet
) -> list[tuple[str, Target]]:
    """Targets that no indexed chunk matches. These questions can never be
    answered, which usually means the index is of a different repository
    version than the question set was written for."""
    by_file: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_file[chunk.file_path].append(chunk)
    missing = []
    for question in question_set.questions:
        for target in question.relevant:
            if not any(target.matches(c) for c in by_file.get(target.file, [])):
                missing.append((question.id, target))
    return missing


class Searcher(Protocol):
    def search(self, query: str, top_k: int = ..., mode: str = ...) -> list: ...


@dataclass
class Scores:
    recall: dict[int, float]
    mrr: float
    count: int


@dataclass
class ModeReport:
    mode: str
    ks: tuple[int, ...]
    # Question id -> rank of the first relevant result (None if not found).
    ranks: dict[str, int | None]
    overall: Scores
    by_category: dict[str, Scores] = field(default_factory=dict)


def score(ranks: list[int | None], ks: tuple[int, ...]) -> Scores:
    n = len(ranks)
    if n == 0:
        return Scores(recall={k: 0.0 for k in ks}, mrr=0.0, count=0)
    recall = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    return Scores(recall=recall, mrr=mrr, count=n)


def evaluate(
    searcher: Searcher,
    question_set: QuestionSet,
    mode: str,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> ModeReport:
    ks = tuple(sorted(set(ks)))
    depth = ks[-1]
    ranks: dict[str, int | None] = {}
    for question in question_set.questions:
        results = searcher.search(question.query, top_k=depth, mode=mode)
        ranks[question.id] = first_relevant_rank(
            [result.chunk for result in results], question.relevant
        )

    categories: dict[str, list[int | None]] = defaultdict(list)
    for question in question_set.questions:
        categories[question.category].append(ranks[question.id])

    return ModeReport(
        mode=mode,
        ks=ks,
        ranks=ranks,
        overall=score(list(ranks.values()), ks),
        by_category={name: score(r, ks) for name, r in sorted(categories.items())},
    )


def report_to_dict(report: ModeReport) -> dict:
    def scores(s: Scores) -> dict:
        return {
            "count": s.count,
            "mrr": round(s.mrr, 4),
            **{f"recall@{k}": round(v, 4) for k, v in s.recall.items()},
        }

    return {
        "mode": report.mode,
        "overall": scores(report.overall),
        "by_category": {name: scores(s) for name, s in report.by_category.items()},
        "ranks": report.ranks,
    }
