"""Build context from retrieved chunks and produce a cited answer via Ollama."""

from __future__ import annotations

from dataclasses import dataclass

from .ollama_client import OllamaClient
from .retriever import Retriever, SearchResult

# Rough cap on how much context text we send to the model. Lower-ranked
# chunks are truncated first if the total would exceed this.
MAX_CONTEXT_CHARS = 12_000

# A truncated block shorter than this isn't worth sending.
_MIN_TRUNCATED_BLOCK_CHARS = 200

_TRUNCATION_MARKER = "\n...[truncated]"


@dataclass
class AnswerResult:
    answer: str
    sources: list[str]
    context: str


def _format_block(index: int, result: SearchResult) -> str:
    chunk = result.chunk
    return (
        f"[SOURCE {index}]\n"
        f"File:\n{chunk.file_path}\n\n"
        f"Symbol:\n{chunk.symbol_name or '(none)'}\n\n"
        f"Lines:\n{chunk.start_line}-{chunk.end_line}\n\n"
        f"Content:\n\n{chunk.content}\n"
    )


def select_context(
    results: list[SearchResult], max_chars: int = MAX_CONTEXT_CHARS
) -> tuple[str, list[SearchResult]]:
    """Build the context string and return it together with the results that
    actually made it into the context (so cited sources match what the model
    saw). Higher-ranked results win; the last one that doesn't fit is
    truncated if a useful amount of it fits, and everything after is dropped.
    """
    separator = "\n"
    parts: list[str] = []
    included: list[SearchResult] = []
    used = 0
    for result in results:
        block = _format_block(len(parts) + 1, result)
        sep_len = len(separator) if parts else 0
        remaining = max_chars - used - sep_len
        if len(block) > remaining:
            budget = remaining - len(_TRUNCATION_MARKER)
            if budget >= _MIN_TRUNCATED_BLOCK_CHARS or (not parts and budget > 0):
                # Always keep at least part of the top-ranked result.
                parts.append(block[:budget] + _TRUNCATION_MARKER)
                included.append(result)
            break
        parts.append(block)
        included.append(result)
        used += sep_len + len(block)
    return separator.join(parts), included


def build_context(results: list[SearchResult], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    return select_context(results, max_chars)[0]


def format_sources(results: list[SearchResult]) -> list[str]:
    return [
        f"{r.chunk.file_path}:{r.chunk.start_line}-{r.chunk.end_line}" for r in results
    ]


def answer_question(
    question: str,
    retriever: Retriever,
    ollama_client: OllamaClient,
    top_k: int = 6,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    mode: str = "hybrid",
) -> AnswerResult:
    results = retriever.search(question, top_k=top_k, mode=mode)
    context, included = select_context(results, max_context_chars)
    answer = ollama_client.ask(question, context)
    return AnswerResult(answer=answer, sources=format_sources(included), context=context)
