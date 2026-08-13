from __future__ import annotations

from pathlib import Path

from pii_detection.registry.folder_rules import ApplyRulesResult, match_activities
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.types import AssociationSource, FolderRule


def _rule(prefix: str, *activity_ids: str) -> FolderRule:
    return FolderRule(prefix=prefix, activity_ids=list(activity_ids))


def test_matches_at_segment_boundary() -> None:
    rule = _rule("HR")
    assert rule.matches("HR")  # the folder itself as a document id
    assert rule.matches("HR/contratti/mario.pdf")
    assert not rule.matches("HRoom/x.pdf")  # not a false prefix
    assert not rule.matches("payroll/HR/x.pdf")  # prefix, not substring


def test_empty_prefix_matches_everything() -> None:
    rule = _rule("")
    assert rule.matches("anything/at/all.pdf")
    assert rule.matches("x.pdf")


def test_match_activities_unions_and_dedups_in_order() -> None:
    rules = [
        _rule("HR", "payroll"),
        _rule("HR/contratti", "contracts", "payroll"),  # overlaps + duplicate id
        _rule("legal", "legal"),  # does not match
    ]
    assert match_activities("HR/contratti/x.pdf", rules) == ["payroll", "contracts"]


def test_match_activities_empty_when_no_rule_matches() -> None:
    assert match_activities("finance/x.pdf", [_rule("HR", "payroll")]) == []


def test_apply_rules_result_defaults_to_zero() -> None:
    result = ApplyRulesResult()
    assert (result.associated, result.skipped_manual, result.unmatched) == (0, 0, 0)

def _registry(tmp_path: Path) -> PIIRepository:
    return PIIRepository(f"sqlite:///{tmp_path / 'pii.db'}")


def test_apply_rules_associates_matching_documents(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record_scan("HR/contratti/mario.pdf", [])
    registry.record_scan("finance/report.pdf", [])
    registry.save_rule("HR", ["payroll"])

    result = registry.apply_folder_rules()

    assert (result.associated, result.unmatched, result.skipped_manual) == (1, 1, 0)
    doc = registry.get_document("HR/contratti/mario.pdf")
    assert doc is not None
    assert doc.activity_ids == ["payroll"]
    assert doc.association_source is AssociationSource.RULE


def test_apply_rules_skips_manual_association(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record_scan("HR/mario.pdf", [])
    registry.assign_activities("HR/mario.pdf", ["chosen_by_hand"])  # MANUAL

    registry.save_rule("HR", ["payroll"])
    result = registry.apply_folder_rules()

    assert result.skipped_manual == 1
    doc = registry.get_document("HR/mario.pdf")
    assert doc is not None
    assert doc.activity_ids == ["chosen_by_hand"]  # manual wins
    assert doc.association_source is AssociationSource.MANUAL


def test_save_rule_upserts_and_normalizes_prefix(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.save_rule("HR/", ["payroll"])       # trailing slash normalized away
    registry.save_rule("HR", ["payroll", "hr"])  # same rule, updated in place

    rules = registry.folder_rules()
    assert len(rules) == 1
    assert rules[0].prefix == "HR"
    assert rules[0].activity_ids == ["payroll", "hr"]


def test_delete_rule_removes_it(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.save_rule("HR", ["payroll"])
    registry.delete_rule("HR")
    assert registry.folder_rules() == []


def test_delete_rule_clears_the_associations_it_had_derived(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record_scan("HR/mario.pdf", [])
    registry.save_rule("HR", ["payroll"])
    registry.apply_folder_rules()  # doc is now RULE-associated to payroll

    result = registry.delete_rule("HR")

    assert result.cleared == 1
    doc = registry.get_document("HR/mario.pdf")
    assert doc is not None
    assert doc.activity_ids == []  # no dangling association left behind
    assert doc.association_source is None


def test_delete_rule_preserves_manual_associations(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record_scan("HR/mario.pdf", [])
    registry.assign_activities("HR/mario.pdf", ["chosen_by_hand"])  # MANUAL
    registry.save_rule("HR", ["payroll"])

    result = registry.delete_rule("HR")

    assert result.skipped_manual == 1 and result.cleared == 0
    doc = registry.get_document("HR/mario.pdf")
    assert doc is not None
    assert doc.activity_ids == ["chosen_by_hand"]  # manual survives the delete
    assert doc.association_source is AssociationSource.MANUAL


def test_apply_reassociates_when_only_some_rules_remain(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record_scan("HR/contratti/mario.pdf", [])
    registry.save_rule("HR", ["payroll"])
    registry.save_rule("HR/contratti", ["contracts"])
    registry.apply_folder_rules()  # doc inherits both payroll and contracts

    registry.delete_rule("HR")  # reconciles: only HR/contratti remains

    doc = registry.get_document("HR/contratti/mario.pdf")
    assert doc is not None
    assert doc.activity_ids == ["contracts"]  # payroll dropped, contracts kept
    assert doc.association_source is AssociationSource.RULE