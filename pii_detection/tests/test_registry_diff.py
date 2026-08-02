"""Tests for the pure scan-diff (block B5, Step 2) — no database involved."""

from __future__ import annotations

from pii_detection.detection.types import (
    ConfirmationLevel,
    DetectionProvenance,
    DetectorKind,
    PIIMatch,
    TextSpan,
)
from pii_detection.registry.diff import diff_scan
from pii_detection.registry.types import PIIInstance


def _inst(start: int, end: int, pii_type: str) -> PIIInstance:
    return PIIInstance(
        document_id="d",
        pii_type=pii_type,
        start=start,
        end=end,
        confidence=0.9,
        confirmation_level="single_source",
        sources=[],
    )


def _match(start: int, end: int, pii_type: str) -> PIIMatch:
    provenance = DetectionProvenance("det.x", DetectorKind.REGEX, pii_type, 0.9)
    return PIIMatch(
        span=TextSpan(start, end),
        text="?",
        pii_type=pii_type,
        confidence=0.9,
        confirmation_level=ConfirmationLevel.SINGLE_SOURCE,
        sources=[provenance],
        document_id="d",
    )


def test_bootstrap_every_match_is_new() -> None:
    diff = diff_scan([], [_match(0, 10, "iban"), _match(20, 30, "email")])
    assert len(diff.new) == 2
    assert not diff.confirmed and not diff.moved and not diff.removed


def test_exact_position_same_type_is_confirmed() -> None:
    diff = diff_scan([_inst(0, 10, "iban")], [_match(0, 10, "iban")])
    assert len(diff.confirmed) == 1
    assert not diff.new and not diff.moved and not diff.removed


def test_same_type_shifted_is_moved() -> None:
    diff = diff_scan([_inst(0, 10, "iban")], [_match(5, 15, "iban")])
    assert len(diff.moved) == 1
    assert not diff.confirmed and not diff.new and not diff.removed


def test_unmatched_match_is_new_and_unmatched_instance_removed() -> None:
    diff = diff_scan([_inst(0, 10, "iban")], [_match(40, 50, "phone")])
    assert [m.pii_type for m in diff.new] == ["phone"]
    assert [i.pii_type for i in diff.removed] == ["iban"]
    assert not diff.confirmed and not diff.moved


def test_same_position_different_type_does_not_match() -> None:
    diff = diff_scan([_inst(0, 10, "iban")], [_match(0, 10, "email")])
    assert [m.pii_type for m in diff.new] == ["email"]
    assert [i.pii_type for i in diff.removed] == ["iban"]


def test_moved_pairs_by_nearest_within_the_same_type() -> None:
    existing = [_inst(0, 10, "iban"), _inst(100, 110, "iban")]
    matches = [_match(2, 12, "iban"), _match(102, 112, "iban")]
    diff = diff_scan(existing, matches)
    assert len(diff.moved) == 2 and not diff.new and not diff.removed
    assert {(inst.start, m.span.start) for inst, m in diff.moved} == {(0, 2), (100, 102)}
