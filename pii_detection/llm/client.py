"""Shared client to a local LLM runtime (Ollama) — used by every AI task.

The selective-AI steps of the system (the ROPA category mapper in B1, later the
sampled detector in B4 and the structure hints in B2) all talk to the same small,
local, CPU-friendly model. To keep that **single** (DRY), they go through one
:class:`LLMClient`: an Adapter over the ``ollama`` Python client whose model and
host come from the environment, so "one model vs one per task" is a configuration
choice, not a code change (see ``doc/plans/ai-assessment.md``).

The underlying backend is injectable (Dependency Injection): tests pass a fake
that satisfies :class:`ChatBackend`, so nothing here needs a running Ollama. The
real backend is created lazily, so importing this module never requires the
optional ``[llm]`` dependency to be installed.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, cast

#: Small quantized model runnable on CPU (thesis choice, §sec:scelta-modello-llm);
#: overridable via the ROPA_LLM_MODEL env var.
DEFAULT_MODEL = "phi4-mini"


class ChatBackend(Protocol):
    """Minimal shape of the chat backend the client drives (the ``ollama.Client``).

    Structural contract, so a test fake satisfies it without importing Ollama.
    """

    def chat(
        self, *, model: str, messages: list[dict[str, str]], options: dict[str, float]
    ) -> Any:
        """Send a chat conversation and return the runtime's raw response.

        :param model: name of the model to run.
        :param messages: ``{"role", "content"}`` turns, in order.
        :param options: runtime options (e.g. ``{"temperature": 0.0}``).
        :returns: a response supporting ``response["message"]["content"]``.
        """
        ...


def _default_backend(host: str | None) -> ChatBackend:
    """Build the real Ollama backend, importing the optional client lazily.

    :param host: Ollama server URL, or ``None`` for the client's own default.
    :returns: an ``ollama.Client`` adapted to the :class:`ChatBackend` shape.
    """
    import ollama

    return cast(ChatBackend, ollama.Client(host=host))


class LLMClient:
    """Adapter over a local LLM runtime, configured from the environment.

    :ivar model: the model name the client runs.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        *,
        temperature: float = 0.0,
        client: ChatBackend | None = None,
    ) -> None:
        """Configure the model and open (or accept) the backend.

        :param model: model name; defaults to the ``ROPA_LLM_MODEL`` environment
            variable, then to :data:`DEFAULT_MODEL`.
        :param host: Ollama server URL; defaults to the ``OLLAMA_HOST``
            environment variable, then to the client's own default. Ignored when
            ``client`` is given.
        :param temperature: sampling temperature; ``0.0`` (default) makes the
            output deterministic and conservative, which the extraction tasks
            (category mapping, detection) rely on for reproducibility.
        :param client: an explicit backend to drive (Dependency Injection); when
            ``None`` a real Ollama client is created lazily.
        """
        self.model = model or os.environ.get("ROPA_LLM_MODEL") or DEFAULT_MODEL
        self._options: dict[str, float] = {"temperature": temperature}
        resolved_host = host if host is not None else os.environ.get("OLLAMA_HOST")
        self._backend = client if client is not None else _default_backend(resolved_host)

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Run one prompt and return the model's text answer.

        :param prompt: the user prompt.
        :param system: optional system instruction prepended to the conversation.
        :returns: the model's answer as text.
        """
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._backend.chat(model=self.model, messages=messages, options=self._options)
        return str(response["message"]["content"])


__all__ = ["DEFAULT_MODEL", "ChatBackend", "LLMClient"]
