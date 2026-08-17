"""Corpus-wide view of the retention axis — block B7 towards B8.

:func:`~pii_detection.compliance.checker.check_document` answers "is *this*
document compliant?", which is the wrong shape for the question a DPO actually
asks of a file share: *show me everything kept past its term*. Opening three
hundred documents one by one to find the dozen that matter is not an operable
procedure, so this module runs the same verdict over the whole registry and keeps
only the rows that say something, ordered by how serious they are.

It adds no rule of its own: the logic is
:func:`~pii_detection.compliance.checker.build_report`, called once per document
and never persisting anything (a read-only view must not write coverage as a side
effect of being looked at). The activities are loaded **once** into a lookup, so
the cost is one read of the ROPA rather than one per document.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pii_detection.compliance.checker import build_report
from pii_detection.compliance.types import RetentionFlag
from pii_detection.registry.repository import PIIRepository
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import ProcessingActivity


@dataclass(frozen=True)
class RetentionRow:
    """What the retention axis has to say about one document.

    :ivar document_id: the document, as the scan identifies it (its relative path).
    :ivar path: original path on disk, when known.
    :ivar reference_date: the date the document is assumed to date from.
    :ivar reference_date_source: provenance of that date
        (:class:`~pii_detection.extraction.dates.DateSource` value); ``file_mtime``
        marks a weak estimate the DPO should treat as an indication, not a finding.
    :ivar activity_ids: the activities the document is associated with (B6).
    :ivar flags: the retention breaches found on it.
    :ivar unresolved: macro-categories present in the document whose retention the
        register states as a criterion, so no comparison was possible.
    :ivar worst_overdue_months: the largest excess among :attr:`flags`, ``0`` when
        the row only carries unverifiable cases.
    """

    document_id: str
    path: str | None
    reference_date: datetime | None
    reference_date_source: str | None
    activity_ids: tuple[str, ...]
    flags: tuple[RetentionFlag, ...]
    unresolved: tuple[str, ...]
    worst_overdue_months: int


def retention_overview(
    *,
    ropa: ROPARepository,
    registry: PIIRepository,
    include_proposed: bool = True,
    now: datetime | None = None,
) -> list[RetentionRow]:
    """Run the retention check over every associated document in the registry.

    Documents with no activity association (B6) are skipped rather than reported:
    with no declared activity there is no declared retention to compare against,
    and listing them here would drown the real breaches in noise. Documents whose
    verdict raises neither a breach nor an unverifiable case produce no row.

    :param ropa: the ROPA repository (declared side); read once.
    :param registry: the detected-PII registry (detected side); **not** written to.
    :param include_proposed: count ``PROPOSED`` category mappings too; ``True`` by
        default, like the dashboard, so the view is useful before the DPO has
        confirmed every mapping.
    :param now: current time, injectable for deterministic results; defaults to the
        current UTC time.
    :returns: the rows worth showing, worst overdue first, then by document id.
    """
    activities: dict[str, ProcessingActivity] = {
        activity.id: activity for activity in ropa.load()
    }

    rows: list[RetentionRow] = []
    for document in registry.documents():
        if not document.activity_ids:
            continue
        resolved = [
            activities[activity_id]
            for activity_id in document.activity_ids
            if activity_id in activities
        ]
        unknown = [
            activity_id
            for activity_id in document.activity_ids
            if activity_id not in activities
        ]
        report = build_report(
            document,
            resolved,
            unknown,
            registry.instances_for(document.document_id),
            include_proposed=include_proposed,
            now=now,
        ).report
        if not report.retention_flags and not report.retention_unresolved:
            continue
        rows.append(
            RetentionRow(
                document_id=document.document_id,
                path=document.path,
                reference_date=document.reference_date,
                reference_date_source=document.reference_date_source,
                activity_ids=tuple(document.activity_ids),
                flags=report.retention_flags,
                unresolved=report.retention_unresolved,
                worst_overdue_months=max(
                    (flag.overdue_months for flag in report.retention_flags), default=0
                ),
            )
        )

    rows.sort(key=lambda row: (-row.worst_overdue_months, row.document_id))
    return rows


def format_overview(rows: list[RetentionRow]) -> str:
    """Render the retention overview as human-readable text (for the CLI).

    References only — no PII value ever appears.

    :param rows: the rows to render, already ordered.
    :returns: a multi-line string.
    """
    if not rows:
        return "No retention issue found across the registry."

    breaches = sum(1 for row in rows if row.flags)
    lines = [
        f"Retention overview: {len(rows)} documents to look at "
        f"({breaches} past their term).",
        "",
    ]
    for row in rows:
        source = row.reference_date_source or "unknown"
        date = row.reference_date.date().isoformat() if row.reference_date else "?"
        lines.append(f"{row.document_id}  [{', '.join(row.activity_ids)}]")
        lines.append(f"  document date: {date} (from {source})")
        for flag in row.flags:
            lines.append(
                f"  OVERDUE +{flag.overdue_months}mo — '{flag.category}': "
                f"{', '.join(flag.pii_types)} present, "
                f"age ~{flag.age_months}mo > {flag.retention_months}mo"
            )
        for category in row.unresolved:
            lines.append(f"  NOT VERIFIABLE — '{category}': retention stated as a criterion")
    return "\n".join(lines)


__all__ = ["RetentionRow", "retention_overview", "format_overview"]
