from rdkit_rag.answering import select_context
from rdkit_rag.models import Chunk
from rdkit_rag.ollama_client import (
    OllamaClient,
    OllamaModelNotFoundError,
    strip_thinking,
)
from rdkit_rag.retriever import SearchResult


def _result(i: int, size: int) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=str(i),
            file_path=f"f{i}.cpp",
            language="C++",
            symbol_name=f"s{i}",
            start_line=1,
            end_line=2,
            content="x" * size,
        ),
        score=1.0 - i / 10,
    )


def test_context_respects_limit_and_sources_match_included_chunks():
    results = [_result(i, 1000) for i in range(10)]
    context, included = select_context(results, max_chars=2500)
    assert len(context) <= 2500
    assert 0 < len(included) < len(results)
    # Every included source must actually appear in the context.
    for r in included:
        assert r.chunk.file_path in context
    assert results[len(included)].chunk.file_path not in context


def test_first_chunk_is_truncated_rather_than_exceeding_limit():
    context, included = select_context([_result(0, 50_000)], max_chars=1000)
    assert len(context) <= 1000
    assert len(included) == 1
    assert context.endswith("[truncated]")


def test_strip_thinking():
    assert strip_thinking("<think>\nhmm\n</think>\n\nAnswer.") == "Answer."


class _FakeOllama(OllamaClient):
    def __init__(self, model, installed):
        super().__init__(model)
        self._installed = installed

    def list_models(self):
        return self._installed


def test_model_check_requires_exact_tag():
    # Previously "qwen3:8b" was considered available when only "qwen3:1.7b"
    # was installed, and the chat request then failed.
    assert not _FakeOllama("qwen3:8b", ["qwen3:1.7b"]).is_available()
    assert _FakeOllama("qwen3:1.7b", ["qwen3:1.7b"]).is_available()
    assert _FakeOllama("llama3", ["llama3:latest"]).is_available()
    try:
        _FakeOllama("qwen3:8b", ["qwen3:1.7b"]).check_available()
    except OllamaModelNotFoundError as e:
        assert "qwen3:1.7b" in str(e)
    else:
        raise AssertionError("expected OllamaModelNotFoundError")


def test_tiny_limit_still_keeps_part_of_top_result():
    context, included = select_context([_result(0, 1000)], max_chars=150)
    assert context and len(context) <= 150
    assert len(included) == 1


def test_registry_model_names_with_port():
    assert _FakeOllama("registry.local:5000/qwen", ["registry.local:5000/qwen:latest"]).is_available()
