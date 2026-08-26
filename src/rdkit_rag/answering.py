"""Build context from retrieved chunks and produce a cited answer via Ollama."""

from __future__ import annotations

from dataclasses import dataclass

from .ollama_client import OllamaClient
from .retriever import Retriever, SearchResult

# Rough cap on how much context text we send to the model. Lower-ranked
# chunks are truncated first if the total would exceed this.
MAX_CONTEXT_CHARS = 12_000


@dataclass
class AnswerResult:
    answer: str
    sources: list[str]
    context: str


def build_context(results: list[SearchResult], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    used = 0
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        block = (
            f"[SOURCE {i}]\n"
            f"File:\n{chunk.file_path}\n\n"
            f"Symbol:\n{chunk.symbol_name or '(none)'}\n\n"
            f"Lines:\n{chunk.start_line}-{chunk.end_line}\n\n"
            f"Content:\n\n{chunk.content}\n"
        )
        if used + len(block) > max_chars and parts:
            # Truncate lower-ranked chunks first rather than dropping the
            # whole block, so at least some content survives.
            remaining = max(max_chars - used, 0)
            if remaining > 200:
                block = block[:remaining] + "\n...[truncated]"
                parts.append(block)
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def format_sources(results: list[SearchResult]) -> list[str]:
    return [
        f"{r.chunk.file_path}:{r.chunk.start_line}-{r.chunk.end_line}" for r in results
    ]


def answer_question(
    question: str,
    retriever: Retriever,
    ollama_client: OllamaClient,
    top_k: int = 6,
) -> AnswerResult:
    results = retriever.search(question, top_k=top_k)
    context = build_context(results)
    answer = ollama_client.ask(question, context)
    sources = format_sources(results)
    return AnswerResult(answer=answer, sources=sources, context=context)
