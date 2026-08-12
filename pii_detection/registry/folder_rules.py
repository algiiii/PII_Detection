"""Folder -> activity association rules — block B6.

The seamless alternative to associating documents one by one. A
:class:`~pii_detection.registry.types.FolderRule` maps a folder prefix to the
processing activities the documents under it belong to; this module holds the
*pure* logic that turns a document id and a set of rules into the activities it
inherits, plus the value object summarizing a bulk application.

The rules live in the registry database and reference ROPA activity ids as
opaque strings (no cross-database foreign key), so the matching here needs
neither the ROPA nor the compliance layer — it is plain, testable data logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pii_detection.registry.types import FolderRule

def match_activities(document_id: str, rules: Sequence[FolderRule]) -> list[str]:
    """Resolve the activities a document inherits from the folder rules.

    Returns the **union** of the activity ids of every rule whose prefix matches
    ``document_id`` (via :meth:`~pii_detection.registry.types.FolderRule.matches`),
    de-duplicated and kept in first-seen order. Unioning across overlapping
    prefixes mirrors the compliance verdict, which already unions the declared
    categories across all of a document's associated activities: a file under both
    ``HR/`` and ``HR/contratti/`` belongs to the activities of both.

    :param document_id: a document id (POSIX path relative to the scan root).
    :param rules: the folder rules to match against.
    :returns: the associated activity ids, unioned and order-preserving; empty if
        no rule matches.
    """

    seen: dict[str, None] = {}
    for rule in rules:
        if rule.matches(document_id):
            for activity_id in rule.activity_ids:
                seen.setdefault(activity_id, None)
    return list(seen)

@dataclass
class ApplyRulesResult:
    """Summary of a bulk application of the folder rules to the registry (B6).

    :ivar associated: documents (re)associated from a matching rule.
    :ivar skipped_manual: documents left untouched because their association was
        set by hand (manual wins over rules).
    :ivar unmatched: documents no rule matched, left as they were.
    """

    associated: int = 0
    skipped_manual: int = 0
    unmatched: int = 0