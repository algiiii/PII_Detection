"""Tests for the custom-patterns extension point (config-driven regex, live).

No Presidio here: the point is that a rule declared in ``custom_patterns.yaml`` is
loaded, validated against the catalog, and detected by the ``RegexDetector`` — the
same detector wired into the live pipeline. So adding a pattern is a YAML edit.
"""

from __future__ import annotations

from pii_detection.detection.config import (
    RegexRuleModel,
    default_config_dir,
    load_category_catalog,
    load_regex_rules,
)
from pii_detection.detection.regex_detector import RegexDetector


def _custom_rules() -> list[RegexRuleModel]:
    base = default_config_dir()
    catalog = load_category_catalog(base / "categories.yaml")
    return list(load_regex_rules(base / "custom_patterns.yaml", catalog))


def test_custom_patterns_file_loads_and_validates() -> None:
    rules = _custom_rules()  # raises if a pattern fails to compile or a pii_type is unknown
    assert any(rule.pii_type == "swiss_avs" for rule in rules)


def test_custom_pattern_is_detected() -> None:
    detector = RegexDetector("regex.custom", _custom_rules())
    candidates = detector.detect("AVS del dipendente: 756.1234.5678.90 — grazie")
    assert any(c.provenance.pii_type == "swiss_avs" for c in candidates)
