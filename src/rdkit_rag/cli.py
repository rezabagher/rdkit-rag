"""CLI entrypoints for rdkit-rag: index, search, ask, eval."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import typer

from .answering import answer_question
from .config import get_config
from .evaluation import (
    QuestionSetError,
    default_questions_path,
    evaluate,
    load_questions,
    missing_targets,
    report_to_dict,
    same_commit,
)
from .indexer import RepositoryNotFoundError, run_index
from .ollama_client import (
    OllamaClient,
    OllamaModelNotFoundError,
    OllamaRequestError,
    OllamaUnavailableError,
)
from .retriever import CorruptIndexError, EmptyIndexError, Retriever
from .store import load_metadata

class SearchMode(str, Enum):
    hybrid = "hybrid"
    vector = "vector"
    keyword = "keyword"


MODE_HELP = (
    "hybrid: combine meaning-based (vector) and keyword (BM25) ranking; "
    "vector: meaning only; keyword: exact words and identifiers only "
    "(fast, no embedding model needed)."
)


def _describe_score(result) -> str:
    ranks = []
    if result.vector_rank is not None:
        ranks.append(f"vector #{result.vector_rank}")
    if result.keyword_rank is not None:
        ranks.append(f"keyword #{result.keyword_rank}")
    suffix = f"  ({', '.join(ranks)})" if ranks else ""
    return f"Score: {result.score:.4g}{suffix}"


app = typer.Typer(
    name="rdkit-rag",
    help="Local RAG for exploring the RDKit codebase.",
    add_completion=False,
)


@app.command()
def index(
    repo_path: Path = typer.Argument(..., help="Path to a local RDKit repository."),
    force: bool = typer.Option(
        False, "--force", help="Re-embed everything (ignore existing embeddings)."
    ),
) -> None:
    """Build the index, or update it after the repository changes.

    Re-running on an existing index only embeds new or changed code, so run
    it again after every git pull.
    """
    config = get_config()
    try:
        result = run_index(repo_path, config, force=force)
    except RepositoryNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo()
    typer.echo(f"Repository: {result.repository_name}")
    if result.repository_commit:
        typer.echo(f"Commit: {result.repository_commit[:12]}")
    typer.echo(f"Files scanned: {result.files_scanned}")
    typer.echo(f"Files indexed: {result.files_indexed}")
    typer.echo(f"Chunks: {result.chunks_created:,}")
    if result.full_rebuild:
        typer.echo(f"Full rebuild: {result.chunks_embedded:,} chunks embedded")
    else:
        typer.echo(
            f"Updated: {result.chunks_embedded:,} new/changed chunks embedded, "
            f"{result.chunks_reused:,} reused"
        )
    typer.echo(f"Embedding model: {result.embedding_model}")
    typer.echo(f"Index location: {result.data_dir}/")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language search query."),
    top_k: int = typer.Option(8, "--top-k", min=1, help="Number of results to return."),
    mode: SearchMode = typer.Option(SearchMode.hybrid, "--mode", help=MODE_HELP),
) -> None:
    """Search the indexed repository with a natural-language query."""
    config = get_config()
    retriever = Retriever(config)
    try:
        results = retriever.search(query, top_k=top_k, mode=mode.value)
    except (EmptyIndexError, CorruptIndexError) as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not results:
        typer.echo("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        typer.echo("-" * 60)
        typer.echo()
        typer.echo(f"{rank}. {_describe_score(result)}")
        typer.echo()
        typer.echo("File:")
        typer.echo(chunk.file_path)
        typer.echo()
        typer.echo("Symbol:")
        typer.echo(chunk.symbol_name or "(none)")
        typer.echo()
        typer.echo("Lines:")
        typer.echo(f"{chunk.start_line}-{chunk.end_line}")
        typer.echo()
        typer.echo("Content:")
        typer.echo()
        typer.echo(chunk.content)
        typer.echo()
    typer.echo("-" * 60)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question."),
    top_k: int = typer.Option(6, "--top-k", min=1, help="Number of chunks to retrieve."),
    mode: SearchMode = typer.Option(SearchMode.hybrid, "--mode", help=MODE_HELP),
    show_context: bool = typer.Option(
        False, "--show-context", help="Print the retrieved context sent to Ollama."
    ),
) -> None:
    """Answer a question about the repository using retrieved code + Ollama."""
    config = get_config()
    ollama_client = OllamaClient(model=config.ollama_model, host=config.ollama_host)

    try:
        ollama_client.check_available()
    except OllamaUnavailableError as e:
        typer.secho("Ollama is not available.", fg=typer.colors.RED)
        typer.echo()
        typer.echo(str(e))
        typer.echo()
        typer.echo("Make sure Ollama is installed and running (e.g. 'ollama serve').")
        typer.echo(f"OLLAMA_HOST={config.ollama_host}")
        raise typer.Exit(code=1)
    except OllamaModelNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        typer.echo()
        typer.echo("Set OLLAMA_MODEL (e.g. in .env) to one of the installed models,")
        typer.echo(f"or pull it with: ollama pull {config.ollama_model}")
        raise typer.Exit(code=1)

    retriever = Retriever(config)
    try:
        result = answer_question(
            question, retriever, ollama_client, top_k=top_k, mode=mode.value
        )
    except (EmptyIndexError, CorruptIndexError, OllamaRequestError) as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if show_context:
        typer.echo("CONTEXT")
        typer.echo()
        typer.echo(result.context)
        typer.echo()

    typer.echo("ANSWER")
    typer.echo()
    typer.echo(result.answer)
    typer.echo()
    typer.echo("SOURCES USED")
    typer.echo()
    for i, source in enumerate(result.sources, start=1):
        typer.echo(f"{i}. {source}")


class EvalMode(str, Enum):
    all = "all"
    hybrid = "hybrid"
    vector = "vector"
    keyword = "keyword"


@app.command("eval")
def eval_command(
    questions: Path | None = typer.Argument(
        None,
        help="Question set (TOML) with the known relevant locations. "
        "Default: the built-in RDKit set.",
    ),
    mode: EvalMode = typer.Option(
        EvalMode.all, "--mode", help="Search mode to evaluate, or 'all' to compare them."
    ),
    k: list[int] = typer.Option(
        [1, 5, 10], "--k", min=1, help="Cut-offs for Recall@k (repeatable)."
    ),
    details: bool = typer.Option(
        False, "--details", help="Show the rank of the first relevant result per question."
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Also write the full results as JSON to this file."
    ),
) -> None:
    """Measure retrieval quality (Recall@k, MRR) on a fixed question set."""
    questions_path = questions if questions is not None else default_questions_path()
    try:
        question_set = load_questions(questions_path)
    except QuestionSetError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Fail on an unusable --output now, not after a possibly long run.
    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            typer.secho(f"Cannot write results to {output}: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    config = get_config()
    retriever = Retriever(config)
    try:
        chunks = retriever.chunks
    except (EmptyIndexError, CorruptIndexError) as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    metadata = load_metadata(config.metadata_path)
    index_commit = metadata.repository_commit if metadata else None
    if (
        question_set.commit
        and index_commit
        and not same_commit(index_commit, question_set.commit)
    ):
        typer.secho(
            f"Note: the index is of commit {index_commit[:12]}, but the question "
            f"set was checked against {question_set.commit[:12]}.",
            fg=typer.colors.YELLOW,
        )
    for qid, target in missing_targets(chunks, question_set):
        where = f"{target.file}" + (f" ({target.symbol})" if target.symbol else "")
        typer.secho(
            f"Warning: {qid}: no indexed chunk matches {where}", fg=typer.colors.YELLOW
        )

    modes = ["hybrid", "vector", "keyword"] if mode is EvalMode.all else [mode.value]
    ks = tuple(sorted(set(k)))
    reports = []
    try:
        for m in modes:
            reports.append(evaluate(retriever, question_set, m, ks))
    except CorruptIndexError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    counts: dict[str, int] = {}
    for q in question_set.questions:
        counts[q.category] = counts.get(q.category, 0) + 1
    summary = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
    typer.echo()
    typer.echo(f"{len(question_set.questions)} questions ({summary})")
    typer.echo()

    header = f"{'mode':<20}" + "".join(f"{'R@' + str(x):>8}" for x in ks) + f"{'MRR':>8}"
    typer.echo(header)
    typer.echo("-" * len(header))

    def row(label: str, scores) -> str:
        cells = "".join(f"{scores.recall[x]:>8.2f}" for x in ks)
        return f"{label:<20}{cells}{scores.mrr:>8.2f}"

    for report in reports:
        typer.echo(row(report.mode, report.overall))
        if len(report.by_category) > 1:
            for name, scores in report.by_category.items():
                typer.echo(row(f"  {name}", scores))
    typer.echo()
    typer.echo(f"MRR is computed over the top {ks[-1]} results.")

    if details:
        typer.echo()
        typer.echo(f"{'question':<30}{'category':<12}" + "".join(f"{m:>9}" for m in modes))
        for q in question_set.questions:
            ranks = "".join(
                f"{(str(r.ranks[q.id]) if r.ranks[q.id] else '-'):>9}" for r in reports
            )
            typer.echo(f"{q.id:<30}{q.category:<12}{ranks}")

    if output is not None:
        payload = {
            "questions": "built-in" if questions is None else str(questions),
            "question_set_commit": question_set.commit,
            "index_commit": index_commit,
            "embedding_model": metadata.embedding_model if metadata else None,
            "ks": list(ks),
            "results": [report_to_dict(r) for r in reports],
        }
        try:
            output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as e:
            typer.secho(f"Cannot write results to {output}: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.echo(f"Results written to {output}")


if __name__ == "__main__":
    app()
