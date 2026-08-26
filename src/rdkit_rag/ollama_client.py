"""Minimal client for a local Ollama server. Ollama is the only LLM
provider supported in this MVP - no cloud APIs, no provider abstraction.
"""

from __future__ import annotations

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


class OllamaUnavailableError(Exception):
    pass


class OllamaClient:
    def __init__(self, model: str, host: str | None = None) -> None:
        self.model = model
        self.host = host

    def _client(self):
        # Imported lazily so commands that don't need Ollama (index, search)
        # don't require the `ollama` package to be importable.
        from ollama import Client

        return Client(host=self.host) if self.host else Client()

    def is_available(self) -> bool:
        try:
            client = self._client()
            models = client.list()
        except Exception:
            return False

        model_names = [m.get("model") or m.get("name") for m in models.get("models", [])]
        # Ollama tags often include a version suffix (e.g. "qwen3:8b"); match
        # both exact and prefix so "qwen3" and "qwen3:8b" both work.
        return any(
            name == self.model or (name and name.split(":")[0] == self.model.split(":")[0])
            for name in model_names
        )

    def ask(self, question: str, context: str) -> str:
        client = self._client()
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
        return response["message"]["content"]
