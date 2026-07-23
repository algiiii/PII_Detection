"""Tests for the LLM-backed category mapper (LLM faked via a backend, no Ollama)."""

from __future__ import annotations

from pii_detection.detection.config import (
    PIICategoryCatalog,
    default_config_dir,
    load_category_catalog,
)
from pii_detection.llm.client import LLMClient
from pii_detection.ropa.ingestion.category_mapper import (
    CategoryMapper,
    DictionaryCategoryMapper,
    LLMCategoryMapper,
    MappedCategory,
)


class _FakeBackend:
    """A ChatBackend returning a fixed answer, for driving the real LLMClient."""

    def __init__(self, content: str) -> None:
        self.content = content

    def chat(self, *, model: str, messages: list[dict[str, str]]) -> dict[str, dict[str, str]]:
        return {"message": {"content": self.content}}


def _catalog() -> PIICategoryCatalog:
    return load_category_catalog(default_config_dir() / "categories.yaml")


def _client(content: str) -> LLMClient:
    return LLMClient(model="m", client=_FakeBackend(content))


def test_llm_mapper_satisfies_protocol() -> None:
    assert isinstance(LLMCategoryMapper(_client("[]"), _catalog()), CategoryMapper)


def test_parses_answer_and_drops_ids_absent_from_catalog() -> None:
    answer = '[{"text": "Bank account details", "pii_types": ["iban", "not_a_type"]}]'
    mapper = LLMCategoryMapper(_client(answer), _catalog())
    assert mapper.map("Bank account details") == [
        MappedCategory("Bank account details", ("iban",))
    ]


def test_extracts_json_array_from_fenced_answer() -> None:
    answer = '```json\n[{"text": "contatti", "pii_types": ["email", "phone"]}]\n```'
    out = LLMCategoryMapper(_client(answer), _catalog()).map("contatti")
    assert out[0].text == "contatti"
    assert set(out[0].pii_types) == {"email", "phone"}


def test_falls_back_to_dictionary_on_unparseable_answer() -> None:
    catalog = _catalog()
    fallback = DictionaryCategoryMapper({"bank account details": ["iban"]}, catalog)
    mapper = LLMCategoryMapper(_client("sorry, I cannot help"), catalog, fallback=fallback)
    assert mapper.map("Bank account details") == [
        MappedCategory("Bank account details", ("iban",))
    ]


def test_without_fallback_returns_raw_text_on_failure() -> None:
    mapper = LLMCategoryMapper(_client("not json"), _catalog())
    assert mapper.map("mystery") == [MappedCategory("mystery", ())]
