"""Value objects for the compliance verdict — block B7.

Pure data, no database access. A :class:`ComplianceReport` is the result of
comparing what a document's associated processing activities *declare* (ROPA, B1)
with what the detection engine *found* (registry, B5).

**Data minimization.** Like the registry it reads, the report holds only
*references* — ``pii_type`` ids, category wordings and counts — and **never** a
PII value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetentionFlag:
    """An approximate retention breach: data of a category kept past its limit.

    Raised when a declared category has a maximum retention, some of its
    ``pii_type`` is still detected in the document, and the document is older than
    that limit. The document age is approximate (from the file's last-modified
    time), so the flag is a signal for the DPO, not a legal determination.

    :ivar activity_id: the activity whose retention rule is exceeded.
    :ivar category: the macro-category wording the retention belongs to.
    :ivar retention_months: the declared maximum retention, in months.
    :ivar age_months: the document's approximate age, in months.
    :ivar pii_types: the detected ``pii_type`` ids of that category still present.
    """

    activity_id: str
    category: str
    retention_months: int
    age_months: int
    pii_types: tuple[str, ...]

    @property
    def overdue_months(self) -> int:
        """How far past its limit the data is kept, in months.

        The severity of the breach: one month over is a reminder, ten years over
        is an incident, and a corpus-wide view has to be able to tell them apart.

        :returns: the excess age over the declared retention.
        """
        return self.age_months - self.retention_months


@dataclass(frozen=True)
class ComplianceReport:
    """The verdict of checking one document against its declared activities (B7).

    :ivar document_id: the checked document.
    :ivar activity_ids: the activities the document is associated with (B6).
    :ivar unknown_activity_ids: associated ids not found in the ROPA database.
    :ivar orphan: detected ``pii_type`` ids declared by *none* of the activities
        — PII present but not accounted for (the core compliance risk).
    :ivar covered: detected ``pii_type`` ids declared by at least one activity.
    :ivar missing: declared ``pii_type`` ids never detected in the document.
    :ivar unresolved: declared categories with no ``pii_type`` (declared but not
        detectable by the engine) — raw wordings, to flag to the DPO.
    :ivar retention_flags: approximate retention breaches.
    :ivar retention_unresolved: wordings of the declared macro-categories whose
        data is present in the document but whose retention is stated as a
        criterion rather than a duration, so no comparison is possible. They are
        **not** breaches: they are the cases nobody checked, surfaced so that
        silence stops looking like compliance.
    """

    document_id: str
    activity_ids: tuple[str, ...]
    unknown_activity_ids: tuple[str, ...]
    orphan: tuple[str, ...]
    covered: tuple[str, ...]
    missing: tuple[str, ...]
    unresolved: tuple[str, ...]
    retention_flags: tuple[RetentionFlag, ...]
    retention_unresolved: tuple[str, ...] = ()

    @property
    def compliant(self) -> bool:
        """Whether the document raises no compliance risk.

        :returns: ``True`` when there is no orphan PII and no retention breach;
            ``missing``/``unresolved`` are informational and do not, by themselves,
            make a document non-compliant.
        """
        return not self.orphan and not self.retention_flags


def format_report(report: ComplianceReport) -> str:
    """Render a compliance report as human-readable text (for the CLI).

    References only — no PII value ever appears.

    :param report: the verdict to render.
    :returns: a multi-line string.
    """

    def _join(items: tuple[str, ...]) -> str:
        return ", ".join(items) if items else "-"

    verdict = "COMPLIANT" if report.compliant else "NON-COMPLIANT"
    lines = [
        f"Compliance report for '{report.document_id}'",
        f"  activities : {_join(report.activity_ids)}",
        f"  verdict    : {verdict}",
        f"  orphan   (detected, not declared): {_join(report.orphan)}",
        f"  covered  (detected and declared) : {_join(report.covered)}",
        f"  missing  (declared, not detected): {_join(report.missing)}",
    ]
    if report.unknown_activity_ids:
        lines.append(f"  unknown activity ids: {_join(report.unknown_activity_ids)}")
    if report.unresolved:
        lines.append(f"  unresolved declared categories: {_join(report.unresolved)}")
    if report.retention_flags:
        lines.append("  retention breaches:")
        for flag in report.retention_flags:
            lines.append(
                f"    - '{flag.category}' [{flag.activity_id}]: "
                f"{_join(flag.pii_types)} present, "
                f"age ~{flag.age_months}mo > {flag.retention_months}mo "
                f"(+{flag.overdue_months}mo)"
            )
    if report.retention_unresolved:
        lines.append(f"  retention not verifiable: {_join(report.retention_unresolved)}")
    return "\n".join(lines)


__all__ = ["RetentionFlag", "ComplianceReport", "format_report"]
