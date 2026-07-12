"""Tests for the config schema and loading (Step 3)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from pii_detection.detection.config import (
    ConfigError,
    PIICategoryCatalog,
    RegexRuleModel,
    load_category_catalog,
    load_detection_config,
    load_ner_labels,
    load_regex_rules,
)

# --- helpers ---------------------------------------------------------------


def _dump(path: Path, data: object) -> Path:
    """Write ``data`` as YAML to ``path`` and return the path."""
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


_TWO_CATEGORIES = {
    "categories": [
        {"id": "email", "label": "Email"},
        {"id": "iban", "label": "IBAN"},
    ]
}


def _catalog(tmp_path: Path) -> PIICategoryCatalog:
    return load_category_catalog(_dump(tmp_path / "categories.yaml", _TWO_CATEGORIES))


# --- shipped config --------------------------------------------------------


class TestDefaultConfig:
    def test_shipped_config_loads(self) -> None:
        config = load_detection_config()
        assert "email" in config.catalog
        assert len(config.regex_rules) > 0
        assert len(config.ner_labels) > 0

    def test_special_category_flag_is_read(self) -> None:
        config = load_detection_config()
        assert config.catalog.require("health_data").special_category is True
        assert config.catalog.require("email").special_category is False

    def test_every_referenced_type_exists(self) -> None:
        config = load_detection_config()
        for rule in config.regex_rules:
            assert rule.pii_type in config.catalog
        for label in config.ner_labels:
            assert label.pii_type in config.catalog


# --- catalog ---------------------------------------------------------------


class TestCatalog:
    def test_duplicate_category_id(self, tmp_path: Path) -> None:
        data = {"categories": [{"id": "email", "label": "A"}, {"id": "email", "label": "B"}]}
        with pytest.raises(ConfigError, match="duplicate category id"):
            load_category_catalog(_dump(tmp_path / "categories.yaml", data))

    def test_require_unknown_raises(self, tmp_path: Path) -> None:
        catalog = _catalog(tmp_path)
        with pytest.raises(ConfigError, match="unknown pii_type"):
            catalog.require("nope")

    def test_get_unknown_returns_none(self, tmp_path: Path) -> None:
        assert _catalog(tmp_path).get("nope") is None
        assert len(_catalog(tmp_path)) == 2


# --- file-level failures ---------------------------------------------------


class TestFileErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_category_catalog(tmp_path / "absent.yaml")

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "categories.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigError, match="empty"):
            load_category_catalog(path)

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "categories.yaml"
        path.write_text("categories: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="malformed YAML"):
            load_category_catalog(path)

    def test_extra_field_forbidden(self, tmp_path: Path) -> None:
        data = {"categories": [{"id": "email", "label": "E", "bogus": 1}]}
        with pytest.raises(ConfigError, match="invalid config"):
            load_category_catalog(_dump(tmp_path / "categories.yaml", data))


# --- regex rules -----------------------------------------------------------


class TestRegexRules:
    def _rules_file(self, tmp_path: Path, rules: list[dict[str, object]]) -> Path:
        return _dump(tmp_path / "regex_rules.yaml", {"rules": rules})

    def test_orphan_pii_type(self, tmp_path: Path) -> None:
        catalog = _catalog(tmp_path)
        rules = self._rules_file(
            tmp_path, [{"rule_id": "r1", "pii_type": "ghost", "pattern": "x"}]
        )
        with pytest.raises(ConfigError, match="not in category catalog"):
            load_regex_rules(rules, catalog)

    def test_duplicate_rule_id(self, tmp_path: Path) -> None:
        catalog = _catalog(tmp_path)
        rules = self._rules_file(
            tmp_path,
            [
                {"rule_id": "dup", "pii_type": "email", "pattern": "a"},
                {"rule_id": "dup", "pii_type": "iban", "pattern": "b"},
            ],
        )
        with pytest.raises(ConfigError, match="duplicate rule_id"):
            load_regex_rules(rules, catalog)

    def test_invalid_regex_pattern(self, tmp_path: Path) -> None:
        catalog = _catalog(tmp_path)
        rules = self._rules_file(
            tmp_path, [{"rule_id": "r1", "pii_type": "email", "pattern": "([a-z"}]
        )
        with pytest.raises(ConfigError, match="invalid config"):
            load_regex_rules(rules, catalog)

    def test_unknown_flag(self, tmp_path: Path) -> None:
        catalog = _catalog(tmp_path)
        rules = self._rules_file(
            tmp_path,
            [{"rule_id": "r1", "pii_type": "email", "pattern": "a", "flags": ["BOGUS"]}],
        )
        with pytest.raises(ConfigError, match="invalid config"):
            load_regex_rules(rules, catalog)

    def test_confidence_out_of_range(self, tmp_path: Path) -> None:
        catalog = _catalog(tmp_path)
        rules = self._rules_file(
            tmp_path,
            [{"rule_id": "r1", "pii_type": "email", "pattern": "a", "base_confidence": 1.5}],
        )
        with pytest.raises(ConfigError, match="invalid config"):
            load_regex_rules(rules, catalog)

    def test_valid_rules_load(self, tmp_path: Path) -> None:
        catalog = _catalog(tmp_path)
        rules = self._rules_file(
            tmp_path,
            [{"rule_id": "r1", "pii_type": "email", "pattern": "a", "flags": ["IGNORECASE"]}],
        )
        loaded = load_regex_rules(rules, catalog)
        assert len(loaded) == 1
        assert loaded[0].rule_id == "r1"

    def test_checksum_fields_are_rejected(self, tmp_path: Path) -> None:
        # Checksum validation is deferred (doc/sviluppi-futuri.md): the old
        # keys must now be rejected as unknown fields, not silently accepted.
        catalog = _catalog(tmp_path)
        rules = self._rules_file(
            tmp_path,
            [{"rule_id": "r1", "pii_type": "email", "pattern": "a", "validator_id": "luhn"}],
        )
        with pytest.raises(ConfigError, match="invalid config"):
            load_regex_rules(rules, catalog)

    def test_re_flags_property(self) -> None:
        rule = RegexRuleModel(
            rule_id="r", pii_type="email", pattern="a", flags=("IGNORECASE", "UNICODE")
        )
        assert rule.re_flags & re.IGNORECASE
        assert rule.re_flags & re.UNICODE


# --- ner labels ------------------------------------------------------------


class TestNerLabels:
    def _labels_file(self, tmp_path: Path, labels: list[dict[str, object]]) -> Path:
        return _dump(tmp_path / "ner_labels.yaml", {"labels": labels})

    def test_orphan_pii_type(self, tmp_path: Path) -> None:
        catalog = _catalog(tmp_path)
        labels = self._labels_file(tmp_path, [{"label": "person", "pii_type": "ghost"}])
        with pytest.raises(ConfigError, match="not in category catalog"):
            load_ner_labels(labels, catalog)

    def test_duplicate_label(self, tmp_path: Path) -> None:
        catalog = _catalog(tmp_path)
        labels = self._labels_file(
            tmp_path,
            [
                {"label": "person", "pii_type": "email"},
                {"label": "person", "pii_type": "iban"},
            ],
        )
        with pytest.raises(ConfigError, match="duplicate label"):
            load_ner_labels(labels, catalog)

    def test_valid_labels_load(self, tmp_path: Path) -> None:
        catalog = _catalog(tmp_path)
        labels = self._labels_file(
            tmp_path, [{"label": "person", "pii_type": "email", "threshold": 0.7}]
        )
        loaded = load_ner_labels(labels, catalog)
        assert loaded[0].threshold == pytest.approx(0.7)
