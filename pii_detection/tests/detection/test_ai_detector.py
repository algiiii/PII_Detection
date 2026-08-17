"""Tests for the generative-AI detector (LLM backend faked, no Ollama needed)."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from pii_detection.detection.ai_detector import (
    AITriggerPolicy,
    LLMDetector,
    build_ai_detector,
)
from pii_detection.detection.config import PIICategoryCatalog, PIICategoryModel
from pii_detection.detection.types import ConfirmationLevel, DetectorKind
from pii_detection.llm.client import LLMClient


class _ScriptedBackend:
    """A :class:`~pii_detection.llm.client.ChatBackend` with scripted answers.

    ``responses`` may be: a fixed ``str`` returned to every call; a callable
    mapping the user-message content to the answer (used to answer differently
    per chunk); or a :class:`BaseException` raised on every call (Ollama down).
    """

    def __init__(self, responses: str | Callable[[str], str] | BaseException) -> None:
        self._responses = responses
        self.calls = 0

    def chat(
        self, *, model: str, messages: list[dict[str, str]], options: dict[str, float]
    ) -> dict[str, dict[str, str]]:
        self.calls += 1
        if isinstance(self._responses, BaseException):
            raise self._responses
        if callable(self._responses):
            content = self._responses(messages[-1]["content"])
        else:
            content = self._responses
        return {"message": {"content": content}}


def _catalog() -> PIICategoryCatalog:
    return PIICategoryCatalog(
        [
            PIICategoryModel(id="person_name", label="Person name"),
            PIICategoryModel(id="iban", label="IBAN"),
        ]
    )


def _detector(
    responses: str | Callable[[str], str] | BaseException,
    *,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> LLMDetector:
    client = LLMClient(model="m", client=_ScriptedBackend(responses))
    return LLMDetector(_catalog(), client, chunk_size=chunk_size, overlap=overlap)


def _payload(*entries: tuple[str, str]) -> str:
    return json.dumps([{"value": v, "pii_type": t} for v, t in entries])


def _present(value: str, pii_type: str = "person_name") -> Callable[[str], str]:
    """Answer with ``value`` only for chunks that contain it, else an empty array."""
    payload = _payload((value, pii_type))
    return lambda user_content: payload if value in user_content else "[]"


# --- span location -----------------------------------------------------------


def test_span_is_located_in_the_full_text() -> None:
    text = "Contact Mario Rossi now"
    matches = _detector(_payload(("Mario Rossi", "person_name"))).detect(text)
    assert len(matches) == 1
    assert (matches[0].span.start, matches[0].span.end) == (8, 19)
    assert matches[0].text == "Mario Rossi"
    assert matches[0].provenance.pii_type == "person_name"
    assert matches[0].provenance.detector_kind is DetectorKind.AI


def test_all_occurrences_of_a_value_become_candidates() -> None:
    text = "Rossi and Rossi"
    matches = _detector(_payload(("Rossi", "person_name"))).detect(text)
    assert [(m.span.start, m.span.end) for m in matches] == [(0, 5), (10, 15)]


# --- anti-hallucination ------------------------------------------------------


def test_unknown_pii_type_is_dropped() -> None:
    text = "Contact Mario Rossi"
    assert _detector(_payload(("Mario Rossi", "unknown_xyz"))).detect(text) == []


def test_value_absent_from_text_is_dropped() -> None:
    text = "Contact Mario Rossi"
    assert _detector(_payload(("Ghost Person", "person_name"))).detect(text) == []


def test_malformed_answer_yields_no_candidates() -> None:
    assert _detector("sorry, I could not find anything").detect("Mario Rossi") == []


def test_json_wrapped_in_prose_and_think_is_parsed() -> None:
    answer = '<think>reasoning here</think> Found: [{"value": "Mario", "pii_type": "person_name"}] done'
    matches = _detector(answer).detect("Mario is here")
    assert len(matches) == 1 and matches[0].text == "Mario"


# --- chunking ----------------------------------------------------------------


def test_second_chunk_reports_a_global_offset() -> None:
    # chunk_size 10, overlap 4 -> step 6; "Neri" sits only past the first window.
    text = "aaaaaaaaaaaaNeribbbb"  # "Neri" at index 12
    matches = _detector(_present("Neri"), chunk_size=10, overlap=4).detect(text)
    assert len(matches) == 1
    assert (matches[0].span.start, matches[0].span.end) == (12, 16)


def test_overlap_duplicates_are_deduplicated() -> None:
    # "John" at index 6 lies in the [6:10] overlap of chunk[0:10] and chunk[6:16].
    text = "aaaaaaJohnbbbbbbbbbb"
    matches = _detector(_present("John"), chunk_size=10, overlap=4).detect(text)
    assert len(matches) == 1
    assert (matches[0].span.start, matches[0].span.end) == (6, 10)


# --- graceful degradation ----------------------------------------------------


def test_unreachable_runtime_yields_no_candidates() -> None:
    assert _detector(ConnectionError("ollama down")).detect("Mario Rossi") == []


# --- trigger policy ----------------------------------------------------------


def test_policy_from_env_unset_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PII_AI_SAMPLING_RATE", raising=False)
    policy = AITriggerPolicy.from_env()
    assert not policy.enabled and not policy.selects(0)


@pytest.mark.parametrize("raw", ["0", "abc", "-5", ""])
def test_policy_from_env_invalid_or_zero_is_disabled(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("PII_AI_SAMPLING_RATE", raw)
    assert AITriggerPolicy.from_env().sampling_rate == 0


def test_policy_from_env_samples_one_in_n(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PII_AI_SAMPLING_RATE", "3")
    policy = AITriggerPolicy.from_env()
    assert policy.enabled
    assert [i for i in range(7) if policy.selects(i)] == [0, 3, 6]


# --- factory -----------------------------------------------------------------


def test_detector_id_embeds_the_env_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PII_LLM_MODEL", "my-model")
    monkeypatch.setattr(
        "pii_detection.llm.client._default_backend", lambda host: _ScriptedBackend("[]")
    )
    assert build_ai_detector().detector_id == "ai.my-model"


def test_confirmation_level_vocabulary_untouched() -> None:
    # Sanity: the detector produces candidates, not merged matches; the AI level
    # is stamped by the merge (Step 2), not here.
    assert ConfirmationLevel.AI_DISCOVERED.value == "ai_discovered"
