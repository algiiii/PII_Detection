"""Compliance check: declared (ROPA) vs detected (registry) — block B7.

The bridge's payoff. Given a document already associated with its processing
activities (B6), compare what those activities *declare* with what the engine
*found*, producing a :class:`~pii_detection.compliance.types.ComplianceReport`.

The block is split in two:

- :func:`build_report` — the **pure** B7 logic over already-loaded domain objects
  (a :class:`~pii_detection.registry.types.Document`, its
  :class:`~pii_detection.ropa.types.ProcessingActivity` and its
  :class:`~pii_detection.registry.types.PIIInstance`), with no I/O, so it is
  testable in isolation and deterministic (the current time is injectable);
- :func:`check_document` — the thin **I/O** orchestration that reads both
  databases (ROPA and registry, joined at application level on the string activity
  id — no physical foreign key), runs :func:`build_report`, and writes the
  per-instance outcome back.

Data minimization holds throughout: the report and the coverage carry only
references, never PII values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from pii_detection.compliance.types import ComplianceReport, RetentionFlag
from pii_detection.extraction.dates import as_utc
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.types import Document, PIIInstance
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import (
    DeclaredMacroCategory,
    MappingState,
    ProcessingActivity,
)

#: Average length of a month in days, for the approximate document-age estimate.
_DAYS_PER_MONTH = 30.44


@dataclass(frozen=True)
class CheckResult:
    """Pure result of the compliance computation.

    :ivar report: the verdict.
    :ivar coverage: instance id → justifying activity id (or ``None`` if orphan),
        to be persisted on the instances.
    """

    report: ComplianceReport
    coverage: dict[int, str | None]


def _macro_types(macro: DeclaredMacroCategory, *, include_proposed: bool) -> set[str]:
    """Union of the ``pii_type`` ids declared under one macro-category.

    The single place where the "which categories count?" rule lives, so the
    coverage comparison and the retention check cannot disagree about it.

    :param macro: the macro-category, with its categories loaded.
    :param include_proposed: also count ``PROPOSED`` mappings; by default only
        DPO-``CONFIRMED`` categories contribute to the authoritative verdict.
    :returns: the set of declared ``pii_type`` ids.
    """
    return {
        pii_type
        for category in macro.categories
        if include_proposed or category.mapping_state is MappingState.CONFIRMED
        for pii_type in category.pii_types
    }


def _declared_types(activity: ProcessingActivity, *, include_proposed: bool) -> set[str]:
    """Union of ``pii_type`` ids declared by an activity's categories.

    :param activity: the processing activity, with its subtree loaded.
    :param include_proposed: also count ``PROPOSED`` mappings; by default only
        DPO-``CONFIRMED`` categories contribute to the authoritative verdict.
    :returns: the set of declared ``pii_type`` ids.
    """
    types: set[str] = set()
    for macro in activity.macro_categories:
        types |= _macro_types(macro, include_proposed=include_proposed)
    return types


def _unresolved_categories(
    activity: ProcessingActivity, *, include_proposed: bool
) -> list[str]:
    """Declared categories that resolve to no ``pii_type`` (declared, not detectable).

    :param activity: the processing activity, with its subtree loaded.
    :param include_proposed: also consider ``PROPOSED`` categories.
    :returns: the raw wordings of the unresolved categories, to flag to the DPO.
    """
    return [
        category.raw_text
        for macro in activity.macro_categories
        for category in macro.categories
        if not category.pii_types
        and (include_proposed or category.mapping_state is MappingState.CONFIRMED)
    ]


def _approx_age_months(reference: datetime, now: datetime) -> int:
    """Approximate age in months between ``reference`` and ``now``.

    Naive ``reference`` values (as SQLite may return) are read as UTC.

    :param reference: the document's reference date (see
        :func:`~pii_detection.extraction.dates.reference_date`).
    :param now: the current time.
    :returns: the whole number of months elapsed, floored at 0.
    """
    days = (now - as_utc(reference)).days
    return max(0, int(days // _DAYS_PER_MONTH))


def build_report(
    document: Document,
    activities: Sequence[ProcessingActivity],
    unknown_activity_ids: Sequence[str],
    instances: Sequence[PIIInstance],
    *,
    include_proposed: bool = False,
    now: datetime | None = None,
) -> CheckResult:
    """Compute the compliance verdict (pure, no I/O).

    :param document: the document being checked (for its ``reference_date``).
    :param activities: the resolved activities the document is associated with, in
        assignment order (an id that could not be resolved is not here).
    :param unknown_activity_ids: associated ids that the ROPA did not resolve.
    :param instances: the document's current (non-removed) PII instances.
    :param include_proposed: count ``PROPOSED`` category mappings too; by default
        only DPO-confirmed ones contribute.
    :param now: current time, injectable for deterministic retention; defaults to
        the current UTC time.
    :returns: the :class:`CheckResult` (report plus the coverage to persist).
    """
    now = now if now is not None else datetime.now(timezone.utc)

    declared_by_activity: dict[str, set[str]] = {
        activity.id: _declared_types(activity, include_proposed=include_proposed)
        for activity in activities
    }
    declared_union: set[str] = set()
    for declared in declared_by_activity.values():
        declared_union |= declared
    detected: set[str] = {instance.pii_type for instance in instances}

    orphan = detected - declared_union
    covered = detected & declared_union
    missing = declared_union - detected

    unresolved: list[str] = []
    for activity in activities:
        unresolved.extend(_unresolved_categories(activity, include_proposed=include_proposed))

    # Per-instance coverage: first associated activity (in order) that declares the
    # instance's type justifies it; none → the instance is orphan.
    coverage: dict[int, str | None] = {}
    for instance in instances:
        if instance.id is None:
            continue
        coverage[instance.id] = next(
            (
                activity.id
                for activity in activities
                if instance.pii_type in declared_by_activity[activity.id]
            ),
            None,
        )

    # Approximate retention. Every declared macro-category whose data is still in
    # the document yields one of two outcomes: a breach, when the document is older
    # than the declared limit, or an "unverifiable" note, when the register states a
    # criterion instead of a duration. The second does not depend on the document's
    # date — it is a gap in the register, not in the document — which is why it is
    # computed outside the date guard: a document whose date is unknown still has
    # categories nobody can check.
    retention_flags: list[RetentionFlag] = []
    retention_unresolved: list[str] = []
    age_months = (
        _approx_age_months(document.reference_date, now)
        if document.reference_date is not None
        else None
    )
    for activity in activities:
        for macro in activity.macro_categories:
            present = _macro_types(macro, include_proposed=include_proposed) & detected
            if not present:
                continue
            if macro.retention_months is None:
                retention_unresolved.append(macro.raw_text)
            elif age_months is not None and age_months > macro.retention_months:
                retention_flags.append(
                    RetentionFlag(
                        activity_id=activity.id,
                        category=macro.raw_text,
                        retention_months=macro.retention_months,
                        age_months=age_months,
                        pii_types=tuple(sorted(present)),
                    )
                )

    report = ComplianceReport(
        document_id=document.document_id,
        activity_ids=tuple(document.activity_ids),
        unknown_activity_ids=tuple(unknown_activity_ids),
        orphan=tuple(sorted(orphan)),
        covered=tuple(sorted(covered)),
        missing=tuple(sorted(missing)),
        unresolved=tuple(unresolved),
        retention_flags=tuple(retention_flags),
        retention_unresolved=tuple(retention_unresolved),
    )
    return CheckResult(report=report, coverage=coverage)


def check_document(
    document_id: str,
    *,
    ropa: ROPARepository,
    registry: PIIRepository,
    include_proposed: bool = False,
    persist_coverage: bool = True,
    now: datetime | None = None,
) -> ComplianceReport:
    """Check a document against its declared activities and persist the outcome (B7).

    Reads the document and its instances from the ``registry`` and the associated
    activities from ``ropa``, runs :func:`build_report`, and (by default) writes
    each instance's justifying activity (or ``None`` if orphan) back to the registry.

    :param document_id: the document to check; must be recorded and associated with
        at least one activity (B6).
    :param ropa: the ROPA repository (declared side).
    :param registry: the detected-PII registry (detected side); the per-instance
        coverage is updated as a side effect unless ``persist_coverage`` is ``False``.
    :param include_proposed: count ``PROPOSED`` category mappings too; by default
        only DPO-confirmed ones contribute.
    :param persist_coverage: write the per-instance coverage back to the registry;
        set ``False`` for a read-only verdict (e.g. rendering a page on a ``GET``).
    :param now: current time, injectable so the retention outcome is deterministic
        end-to-end and not only in :func:`build_report`; defaults to the current
        UTC time.
    :returns: the compliance verdict.
    :raises KeyError: if the document was never recorded.
    :raises ValueError: if the document has no activity association (run B6 first).
    """
    document = registry.get_document(document_id)
    if document is None:
        raise KeyError(document_id)
    if not document.activity_ids:
        raise ValueError(
            f"document {document_id!r} has no activity association (assign it first)"
        )

    activities: list[ProcessingActivity] = []
    unknown: list[str] = []
    for activity_id in document.activity_ids:
        activity = ropa.get(activity_id)
        if activity is None:
            unknown.append(activity_id)
        else:
            activities.append(activity)

    instances = registry.instances_for(document_id)
    result = build_report(
        document,
        activities,
        unknown,
        instances,
        include_proposed=include_proposed,
        now=now,
    )
    if persist_coverage:
        registry.apply_coverage(document_id, result.coverage)
    return result.report


__all__ = ["CheckResult", "build_report", "check_document"]
