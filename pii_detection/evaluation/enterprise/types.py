"""Value objects describing a corpus *before* a single byte hits the disk.

The generator is split in two halves on purpose: planning (pure — what files
exist, where, with which content, size and modification time) and writing
(effectful). Everything worth asserting lives in the plan, so the tests that
matter run in milliseconds and never touch a temporary directory, the same way
:func:`~pii_detection.registry.diff.diff_scan` and
:func:`~pii_detection.registry.folder_rules.match_activities` are pure.

The whole plan is immutable and comparable: generating twice with the same seed
must yield two equal :class:`CorpusPlan` values, which is how reproducibility is
tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:  # avoid a cycle: profiles imports this module
    from pii_detection.evaluation.enterprise.profiles import RetentionExpectation


class SizeClass(StrEnum):
    """How long a document is, in the terms the corpus is balanced by.

    The line counts are targets for the body builders; the renderer lays out
    roughly 30 lines per PDF page, so the classes span "a note" to "a manual".

    :cvar SHORT: 5-15 lines — a note, a ticket, a payment advice (~1 page).
    :cvar MEDIUM: 40-90 lines — a record or a letter (1-2 pages).
    :cvar LONG: 120-260 lines — a contract or a report (3-8 pages).
    :cvar HUGE: 900-1800 lines — a manual or a yearly export (30-60 pages).
    """

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    HUGE = "huge"


class Expectation(StrEnum):
    """What a scan of the corpus must do with a file — asserted, not hoped for.

    :cvar SCANNABLE: supported format and readable: it must be ingested.
    :cvar SKIPPED: unsupported extension: it must land in the skipped list of
        :class:`~pii_detection.registry.scan_folder.FolderPlan`, silently.
    :cvar ERROR: supported extension but unreadable (empty, truncated, wrong
        encoding, an office lock file): it must land in the errors list of
        :class:`~pii_detection.registry.scan_folder.FolderScanResult` **and the
        batch must go on**.
    """

    SCANNABLE = "scannable"
    SKIPPED = "skipped"
    ERROR = "error"


class Profile(StrEnum):
    """Which tree to build.

    :cvar REALISTIC: a plausible company file share, owing nothing to any ROPA —
        the honest simulation, where nobody has tidied the folders to match the
        register.
    :cvar ROPA: a tree anchored to a real ROPA file, with violations planted on
        purpose (orphan PII, expired retention, folders no rule reaches) so the
        association (B6) and the compliance verdict (B7) have something to find.
    """

    REALISTIC = "realistic"
    ROPA = "ropa"


@dataclass(frozen=True)
class FolderSpec:
    """A folder of the tree and what it is allowed to contain.

    Documents are drawn per folder, not globally, so the content matches the
    place: access logs end up under ``IT/Log``, payslips under HR. ``weight`` is
    the relative share of the corpus this folder receives.

    :ivar path: path relative to the scanned root, POSIX (e.g.
        ``"Risorse Umane/Contratti/2023"``).
    :ivar kinds: ids of the document archetypes this folder may hold.
    :ivar formats: file formats the documents may be written in, among
        ``"pdf"``, ``"docx"``, ``"txt"``.
    :ivar weight: relative share of the documents assigned to this folder.
    :ivar year: nominal year of the documents, driving their modification time
        (``None`` = recent). Old folders are what makes the retention check of
        block B7 have something to complain about.
    """

    path: str
    kinds: tuple[str, ...]
    formats: tuple[str, ...]
    weight: int = 1
    year: int | None = None


@dataclass(frozen=True)
class DocumentSpec:
    """One planned file: its identity, its content and its expected fate.

    ``annotated_text`` is the single source of truth of the content: the clean
    text written to the file and the gold record are both *derived* from it in
    one pass, so they cannot drift apart.

    :ivar relative_path: path relative to the scanned root, POSIX. It is
        verbatim the ``document_id`` the recursive scan will compute, which is
        what lets gold, manifest and registry be joined without translation.
    :ivar kind: id of the archetype that produced the body (``"hr_contract"``,
        ``"meeting_minutes"``, …); ``""`` for noise files.
    :ivar size_class: the length band the body was built for.
    :ivar file_format: ``"pdf"``, ``"docx"``, ``"txt"`` or, for noise, the raw
        extension without the dot.
    :ivar annotated_text: the body with ``{{pii_type:value}}`` markers; empty
        for noise files, whose bytes are produced by the noise module instead.
    :ivar modified_at: modification time to stamp on the file, which the registry
        reads back as the document's ``reference_date`` for the retention check
        (these formats carry no internal date, so the file system is the source).
    :ivar expectation: what a scan must do with this file.
    """

    relative_path: str
    kind: str
    size_class: SizeClass
    file_format: str
    annotated_text: str
    modified_at: datetime
    expectation: Expectation


@dataclass(frozen=True)
class CorpusPlan:
    """A whole corpus, decided but not yet written.

    :ivar profile: the profile that produced it.
    :ivar seed: the seed it was planned with (reproducibility).
    :ivar documents: every planned file, noise included, in write order.
    :ivar folder_rules: ``(prefix, activity_ids)`` pairs ready for
        :meth:`~pii_detection.registry.repository.PIIRepository.save_rule`;
        empty outside the ``ropa`` profile.
    :ivar expected_retention: folder prefixes whose documents must come back as
        retention breaches, with the term they exceed — the expectation the
        planted archive folders exist to produce.
    :ivar expected_orphans: ``pii_type`` ids planted in covered folders while no
        associated activity declares them, so the compliance check must report
        them as orphan; empty outside the ``ropa`` profile.
    """

    profile: Profile
    seed: int
    documents: tuple[DocumentSpec, ...]
    folder_rules: tuple[tuple[str, tuple[str, ...]], ...] = ()
    expected_orphans: tuple[str, ...] = ()
    expected_retention: tuple[RetentionExpectation, ...] = ()

    def scannable(self) -> tuple[DocumentSpec, ...]:
        """:returns: the documents a scan must ingest (noise excluded)."""
        return tuple(d for d in self.documents if d.expectation is Expectation.SCANNABLE)


__all__ = [
    "CorpusPlan",
    "DocumentSpec",
    "Expectation",
    "FolderSpec",
    "Profile",
    "SizeClass",
]
