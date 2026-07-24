"""Tests for the shared LLM client (backend faked, no running Ollama needed)."""

from __future__ import annotations

import pytest

from pii_detection.llm.client import DEFAULT_MODEL, LLMClient


class _FakeBackend:
    """A :class:`~pii_detection.llm.client.ChatBackend` that records its calls."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[tuple[str, list[dict[str, str]], dict[str, float]]] = []

    def chat(
        self, *, model: str, messages: list[dict[str, str]], options: dict[str, float]
    ) -> dict[str, dict[str, str]]:
        self.calls.append((model, messages, options))
        return {"message": {"content": self.content}}


def test_complete_builds_messages_and_returns_content() -> None:
    fake = _FakeBackend("resolved")
    client = LLMClient(model="m", client=fake)

    assert client.complete("map this", system="you resolve categories") == "resolved"

    model, messages, _ = fake.calls[0]
    assert model == "m"
    assert messages == [
        {"role": "system", "content": "you resolve categories"},
        {"role": "user", "content": "map this"},
    ]


def test_complete_without_system_sends_only_user_turn() -> None:
    fake = _FakeBackend("x")
    LLMClient(model="m", client=fake).complete("hi")
    assert fake.calls[0][1] == [{"role": "user", "content": "hi"}]


def test_defaults_to_deterministic_temperature() -> None:
    fake = _FakeBackend("x")
    LLMClient(model="m", client=fake).complete("hi")
    assert fake.calls[0][2] == {"temperature": 0.0}


def test_model_comes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROPA_LLM_MODEL", "from-env")
    assert LLMClient(client=_FakeBackend("x")).model == "from-env"


def test_model_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROPA_LLM_MODEL", raising=False)
    assert LLMClient(client=_FakeBackend("x")).model == DEFAULT_MODEL
