"""Minimal client for a local Ollama server. Ollama is the only LLM
provider supported in this MVP - no cloud APIs, no provider abstraction.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = """You are an assistant helping a developer understand the RDKit source code.

Answer primarily using the repository context provided to you.

Rules:

1. Do not invent files, functions, classes, or implementation details.
2. Prefer direct evidence from the retrieved repository context over general knowledge.
3. If something is not clearly supported by the retrieved context, say that it is uncertain.
4. If the retrieved context is insufficient, explicitly say:
"The retrieved repository context is not sufficient to answer this confidently."
5. Cite important technical claims using:
[file_path:start_line-end_line]
6. Keep answers technical and focused.
7. When explaining a process, clearly distinguish what is directly visible in the retrieved code from what is inferred.
"""

# Reasoning models (qwen3, deepseek-r1, ...) may emit their chain of thought
# inline as <think>...</think> depending on the Ollama version. That is not
# part of the answer, so strip it.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaUnavailableError(Exception):
    pass


class OllamaModelNotFoundError(Exception):
    pass


class OllamaRequestError(Exception):
    pass


def _field(obj, name: str):
    """Read a field from either a dict (ollama-python < 0.4) or a pydantic
    response object (ollama-python >= 0.4)."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _normalize_model_name(name: str) -> str:
    """Ollama treats an untagged name as ':latest'. Only the last path
    segment can carry a tag ("registry.local:5000/qwen" is untagged)."""
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class OllamaClient:
    def __init__(self, model: str, host: str | None = None) -> None:
        self.model = model
        self.host = host

    def _client(self):
        # Imported lazily so commands that don't need Ollama (index, search)
        # don't require the `ollama` package to be importable.
        from ollama import Client

        return Client(host=self.host) if self.host else Client()

    def list_models(self) -> list[str]:
        """Return the names of locally available models.

        Raises OllamaUnavailableError if the server can't be reached.
        """
        try:
            response = self._client().list()
        except Exception as e:  # connection refused, timeout, bad host, ...
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.host or 'the default host'}: {e}"
            ) from e

        names = []
        for m in _field(response, "models") or []:
            name = _field(m, "model") or _field(m, "name")
            if name:
                names.append(name)
        return names

    def check_available(self) -> None:
        """Raise a descriptive error if Ollama is down or the configured model
        hasn't been pulled. Matching is exact (after normalizing an untagged
        name to ':latest'): "qwen3:8b" must not be satisfied by "qwen3:1.7b".
        """
        names = self.list_models()
        wanted = _normalize_model_name(self.model)
        if not any(_normalize_model_name(n) == wanted for n in names):
            raise OllamaModelNotFoundError(
                f"Model '{self.model}' is not available in Ollama. "
                f"Installed models: {', '.join(names) or '(none)'}"
            )

    def is_available(self) -> bool:
        try:
            self.check_available()
        except (OllamaUnavailableError, OllamaModelNotFoundError):
            return False
        return True

    def ask(self, question: str, context: str) -> str:
        client = self._client()
        try:
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Repository context:\n\n{context}\n\nQuestion: {question}",
                    },
                ],
            )
        except Exception as e:
            raise OllamaRequestError(f"Ollama request failed: {e}") from e

        message = _field(response, "message")
        content = _field(message, "content") if message is not None else None
        return strip_thinking(content or "")
