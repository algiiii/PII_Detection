"""Delta between a document's current PII and a new scan — block B5, Step 2.

Isolated and pure (no database), so the changelog logic is testable on its own.
Given the current :class:`~pii_detection.registry.types.PIIInstance` of a document
and the fresh :class:`~pii_detection.detection.types.PIIMatch` of a new scan, it
classifies each into the closed vocabulary
:class:`~pii_detection.registry.types.ChangeType`:

- **CONFIRMED** — same ``pii_type`` at the same position (found again unchanged);
- **MOVED** — same ``pii_type`` matched to the nearest previous instance at a
  different position (the pragmatic identity rule for the "moved?" open question:
  no value is stored, so identity is approximated by type + proximity);
- **NEW** — a match with no counterpart among the current instances;
- **REMOVED** — a current instance no counterpart claimed.

Matching is greedy and deterministic: exact positions first (CONFIRMED), then the
closest same-type pairs by start offset (MOVED); the leftovers are NEW / REMOVED.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pii_detection.detection.types import PIIMatch
from pii_detection.registry.types import PIIInstance


@dataclass
class ScanDiff:
    """Classification of a new scan against a document's current instances.

    :ivar confirmed: ``(instance, match)`` pairs unchanged in position.
    :ivar moved: ``(instance, match)`` pairs of the same type at a new position.
    :ivar new: matches with no current counterpart — brand-new instances.
    :ivar removed: current instances no match claimed — no longer present.
    """

    confirmed: list[tuple[PIIInstance, PIIMatch]] = field(default_factory=list)
    moved: list[tuple[PIIInstance, PIIMatch]] = field(default_factory=list)
    new: list[PIIMatch] = field(default_factory=list)
    removed: list[PIIInstance] = field(default_factory=list)


def diff_scan(existing: Sequence[PIIInstance], matches: Sequence[PIIMatch]) -> ScanDiff:
    """Compare the current instances of a document with a new scan's matches.

    Pure function (no I/O). On a first scan (``existing`` empty) every match is
    :attr:`~pii_detection.registry.types.ScanDiff.new`, which is the bootstrap.

    :param existing: the document's current (non-removed) instances.
    :param matches: the matches produced by the new scan.
    :returns: the classified :class:`ScanDiff`.
    """
    result = ScanDiff()
    used_existing: set[int] = set()
    used_match: set[int] = set()

    # Pass 1 — exact position and same pii_type: CONFIRMED.
    for match_index, match in enumerate(matches):
        for instance_index, instance in enumerate(existing):
            if instance_index in used_existing:
                continue
            if (
                instance.pii_type == match.pii_type
                and instance.start == match.span.start
                and instance.end == match.span.end
            ):
                result.confirmed.append((instance, match))
                used_existing.add(instance_index)
                used_match.add(match_index)
                break

    # Pass 2 — same pii_type, nearest by start offset: MOVED (greedy).
    candidates: list[tuple[int, int, int]] = []  # (distance, match_index, instance_index)
    for match_index, match in enumerate(matches):
        if match_index in used_match:
            continue
        for instance_index, instance in enumerate(existing):
            if instance_index in used_existing or instance.pii_type != match.pii_type:
                continue
            candidates.append((abs(instance.start - match.span.start), match_index, instance_index))
    for _distance, match_index, instance_index in sorted(candidates):
        if match_index in used_match or instance_index in used_existing:
            continue
        result.moved.append((existing[instance_index], matches[match_index]))
        used_match.add(match_index)
        used_existing.add(instance_index)

    result.new = [m for i, m in enumerate(matches) if i not in used_match]
    result.removed = [e for i, e in enumerate(existing) if i not in used_existing]
    return result


__all__ = ["ScanDiff", "diff_scan"]
