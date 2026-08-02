"""Associate a document with its processing activities — block B6.

Explicit, DPO-driven association behind a Strategy Protocol
(:class:`ActivityAssigner`), so an automatic (content-based) or AI proposer can be
slotted in later without touching the compliance check (B7) or the CLI. The only
strategy implemented in this slice is the explicit one: the DPO provides the ids.

The association is stored on the registry side
(:attr:`~pii_detection.registry.types.Document.activity_ids`); the ids reference
:class:`~pii_detection.ropa.types.ProcessingActivity` in the separate ROPA
database, joined at application level (no cross-database foreign key).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pii_detection.registry.repository import PIIRepository


@runtime_checkable
class ActivityAssigner(Protocol):
    """Shape of a strategy proposing the activities a document belongs to.

    Structural typing: B6 depends only on ``assign(document_id, detected_types) ->
    list[str]``. The explicit strategy returns what the DPO passed in; a future
    content-based or AI proposer implements the same contract and drops in through
    :func:`persist_assignment` unchanged.
    """

    def assign(self, document_id: str, detected_types: Sequence[str]) -> list[str]:
        """Return the activity ids the document should be associated with.

        :param document_id: identifier of the document.
        :param detected_types: the ``pii_type`` ids detected in it, for strategies
            that propose from content (ignored by the explicit strategy).
        :returns: the associated activity ids.
        """
        ...


class ExplicitAssigner:
    """The DPO-provided association — the human-in-the-loop baseline.

    Returns exactly the ids the DPO assigned, ignoring the document content. It is
    the honest basis for the compliance check: the activities are chosen by a human,
    not inferred to minimize orphans.

    :ivar _activity_ids: the ids the DPO assigned (private).
    """

    def __init__(self, activity_ids: Sequence[str]) -> None:
        """:param activity_ids: the activities the DPO assigns to the document."""
        self._activity_ids = list(activity_ids)

    def assign(self, document_id: str, detected_types: Sequence[str]) -> list[str]:
        """Return the DPO-provided ids, unchanged.

        :param document_id: identifier of the document (unused).
        :param detected_types: detected ``pii_type`` ids (unused).
        :returns: the assigned activity ids.
        """
        return list(self._activity_ids)


def persist_assignment(
    repository: PIIRepository, document_id: str, assigner: ActivityAssigner
) -> list[str]:
    """Run an assignment strategy for a document and persist its result (B6).

    Fetches the document's detected ``pii_type`` ids (so a content-based strategy
    can use them), asks the ``assigner`` for the activities, and stores them via
    :meth:`~pii_detection.registry.repository.PIIRepository.assign_activities`.
    The explicit strategy ignores the detected types; a future proposer does not —
    and this path stays the same.

    :param repository: the registry to read the document from and write to.
    :param document_id: identifier of an already-recorded document.
    :param assigner: the strategy producing the activity ids.
    :returns: the activity ids that were persisted.
    :raises KeyError: if the document was never recorded.
    :raises ValueError: if the strategy yields an empty set or a blank id.
    """
    detected_types = [instance.pii_type for instance in repository.instances_for(document_id)]
    activity_ids = assigner.assign(document_id, detected_types)
    repository.assign_activities(document_id, activity_ids)
    return activity_ids


__all__ = ["ActivityAssigner", "ExplicitAssigner", "persist_assignment"]
