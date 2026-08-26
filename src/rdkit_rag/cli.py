"""CLI entrypoints for rdkit-rag: index, search, ask."""

from __future__ import annotations

from pathlib import Path

import typer

from .answering import answer_question
from .config import get_config
from .indexer import IndexAlreadyExistsError, RepositoryNotFoundError, run_index
from .ollama_client import OllamaClient
from .retriever import EmptyIndexError, Retriever

app = typer.Typer(
    name="rdkit-rag",
    help="Local RAG for exploring the RDKit codebase.",
    add_completion=False,
)


@app.command()
def index(
    repo_path: Path = typer.Argument(..., help="Path to a local RDKit repository."),
    force: bool = typer.Option(
        False, "--force", help="Rebuild the index if one already exists."
    ),
) -> None:
    """Scan, chunk, embed, and store the repository locally."""
    config = get_config()
    try:
        result = run_index(repo_path, config, force=force)
    except RepositoryNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except IndexAlreadyExistsError as e:
        typer.secho(str(e), fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    typer.echo()
    typer.echo(f"Repository: {result.repository_name}")
    typer.echo(f"Files scanned: {result.files_scanned}")
    typer.echo(f"Files indexed: {result.files_indexed}")
    typer.echo(f"Chunks created: {result.chunks_created}")
    typer.echo(f"Embedding model: {result.embedding_model}")
    typer.echo(f"Index location: {result.data_dir}/")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language search query."),
    top_k: int = typer.Option(8, "--top-k", help="Number of results to return."),
) -> None:
    """Search the indexed repository with a natural-language query."""
    config = get_config()
    retriever = Retriever(config)
    try:
        results = retriever.search(query, top_k=top_k)
    except EmptyIndexError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not results:
        typer.echo("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        typer.echo("-" * 60)
        typer.echo()
        typer.echo(f"{rank}. Score: {result.score:.2f}")
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
    top_k: int = typer.Option(6, "--top-k", help="Number of chunks to retrieve."),
    show_context: bool = typer.Option(
        False, "--show-context", help="Print the retrieved context sent to Ollama."
    ),
) -> None:
    """Answer a question about the repository using retrieved code + Ollama."""
    config = get_config()
    ollama_client = OllamaClient(model=config.ollama_model, host=config.ollama_host)

    if not ollama_client.is_available():
        typer.secho("Ollama is not available.", fg=typer.colors.RED)
        typer.echo()
        typer.echo("Make sure Ollama is installed and running.")
        typer.echo()
        typer.echo("Also ensure the configured model is available:")
        typer.echo(f"OLLAMA_MODEL={config.ollama_model}")
        raise typer.Exit(code=1)

    retriever = Retriever(config)
    try:
        result = answer_question(question, retriever, ollama_client, top_k=top_k)
    except EmptyIndexError as e:
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


if __name__ == "__main__":
    app()
