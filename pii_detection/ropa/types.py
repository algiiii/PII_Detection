"""Data model of the ROPA (Record of Processing Activities) — block B1.

Design and rationale in ``doc/sections/5_architettura.tex`` (§``sec:modello-ropa``).
The flat CNIL sheet is normalized around :class:`ProcessingActivity`, gathered by
the :class:`ROPA` container. The model is **technology-agnostic**: it is the
in-memory structure a persistence layer will store, but the storage technology is
deliberately deferred.

Two design choices make the structure fit the compliance-checking purpose (B7)
rather than merely mirroring the source document:

1. **Shared vocabulary (bridge to B4).** :class:`DeclaredDataCategory` resolves a
   declared category onto the same ``pii_type`` catalog used by the detection
   layer, one-to-many, so comparing what the ROPA *declares* with what the engine
   *finds* becomes a set operation. The AI proposes the mapping, the DPO confirms
   it (:class:`MappingState`).
2. **Machine-computable retention.** :class:`Retention` stores a structured
   ``duration_months`` next to the original wording, so the "past retention?"
   check is computable rather than a prose read.

This first version keeps the remaining CNIL fields (data subjects, recipients,
transfers, security measures, information systems) as plain strings; they become
richer entities only once the ingestion actually needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MappingState(str, Enum):
    """Confirmation state of a declared-category → ``pii_type`` mapping.

    :cvar PROPOSED: mapping produced by the selective AI, not yet validated.
    :cvar CONFIRMED: mapping validated by the DPO; safe to use for compliance.
    """

    PROPOSED = "proposed"
    CONFIRMED = "confirmed"


@dataclass(frozen=True)
class DeclaredDataCategory:
    """A ROPA data category resolved onto the shared ``pii_type`` catalog.

    The bridge to the detection layer: the free-text register entry is mapped
    onto zero or more catalog ids. Empty ``pii_types`` means "not mapped yet".

    :ivar raw_text: original free-text category, kept for audit (``"dati anagrafici"``).
    :ivar pii_types: catalog ids it resolves to, one-to-many (``("person_name", ...)``).
    :ivar mapping_state: whether the mapping is AI-proposed or DPO-confirmed.
    """

    raw_text: str
    pii_types: tuple[str, ...] = ()
    mapping_state: MappingState = MappingState.PROPOSED


@dataclass(frozen=True)
class Retention:
    """A retention rule attached to a processing activity.

    :ivar raw_text: original retention wording, kept for audit.
    :ivar duration_months: retention length in months; ``None`` when the register
        states a criterion rather than a fixed duration.
    """

    raw_text: str
    duration_months: int | None = None

    def __post_init__(self) -> None:
        """:raises ValueError: if ``duration_months`` is negative."""
        if self.duration_months is not None and self.duration_months < 0:
            raise ValueError(f"negative duration_months: {self.duration_months}")


@dataclass
class ProcessingActivity:
    """A single processing activity — the central entity of the ROPA.

    Mirrors one CNIL sheet. Mutable so the ingestion (B1) can fill the lists
    incrementally. The plain-string lists hold CNIL fields not yet modeled as
    entities.

    :ivar activity_id: stable identifier of the activity.
    :ivar name: name of the processing activity.
    :ivar purpose: declared purpose.
    :ivar legal_basis: legal basis.
    :ivar controller: data controller.
    :ivar dpo: data protection officer, if designated.
    :ivar data_categories: declared data categories, resolved onto ``pii_type``.
    :ivar retentions: retention rules.
    :ivar data_subjects: categories of data subjects.
    :ivar recipients: recipients of the data.
    :ivar third_country_transfers: transfers outside the EU/EEA.
    :ivar security_measures: technical/organizational measures.
    :ivar information_systems: systems/repositories the data resides in.
    """

    activity_id: str # Simple String
    name: str # Simple String
    purpose: str # Simple String
    legal_basis: str # Simple String
    controller: str # Simple String
    dpo: str | None = None  # Optional Simple String
    data_categories: list[DeclaredDataCategory] = field(default_factory=list) # List of `DeclaredDataCategory` instances, initialized as an empty list by default
    retentions: list[Retention] = field(default_factory=list) # "" Retention "" 
    data_subjects: list[str] = field(default_factory=list) # "" str ""
    recipients: list[str] = field(default_factory=list) # "" str ""
    third_country_transfers: list[str] = field(default_factory=list) # "" str ""
    security_measures: list[str] = field(default_factory=list) # "" str ""
    information_systems: list[str] = field(default_factory=list) # "" str ""

    def declared_pii_types(self, *, confirmed_only: bool = False) -> set[str]:
        """Aggregate the ``pii_type`` ids this activity declares (the "expected"
        set the compliance check compares against the detected one).

        :param confirmed_only: if ``True``, include only DPO-confirmed mappings.
        :returns: the union of ``pii_type`` ids across the data categories.
        """
        return {
            pii_type
            for category in self.data_categories
            if not confirmed_only or category.mapping_state is MappingState.CONFIRMED
            for pii_type in category.pii_types
        }


@dataclass
class ROPA:
    """A Record of Processing Activities: the set of processing activities.

    :ivar activities: the processing activities of the register.
    """

    activities: list[ProcessingActivity] = field(default_factory=list)

    def __post_init__(self) -> None:
        """:raises ValueError: if two activities share the same ``activity_id``."""
        seen: set[str] = set()
        for activity in self.activities:
            if activity.activity_id in seen:
                raise ValueError(f"duplicate activity_id: {activity.activity_id!r}")
            seen.add(activity.activity_id)

    def activity(self, activity_id: str) -> ProcessingActivity | None:
        """Look up an activity by id.

        :param activity_id: identifier to resolve.
        :returns: the matching activity, or ``None`` if absent.
        """
        for activity in self.activities:
            if activity.activity_id == activity_id:
                return activity
        return None

    def declared_pii_types(self, *, confirmed_only: bool = False) -> set[str]:
        """Union of the declared ``pii_type`` ids across every activity.

        :param confirmed_only: if ``True``, include only DPO-confirmed mappings.
        :returns: the aggregated set of declared ``pii_type`` ids.
        """
        result: set[str] = set()
        for activity in self.activities:
            result |= activity.declared_pii_types(confirmed_only=confirmed_only)
        return result


__all__ = [
    "MappingState",
    "DeclaredDataCategory",
    "Retention",
    "ProcessingActivity",
    "ROPA",
]
