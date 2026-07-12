"""Shared data model of the detection layer (block B4).

Defined in ``doc/planning.md`` §"Modello dati comune". The layer distinguishes
three data levels:

1. :class:`PIICandidate` — raw output of ONE detector, pre-merge.
2. :class:`PIIMatch` — output of the merge (the "unified PII" that feed B5).
3. ``IstanzaPII`` / ``VariazionePII`` — persistence (out of scope, in B5/B6).

**Minimization (§2.3.11).** The detection-time DTOs may carry the ``text``
field (needed for the merge and for report readability) and live only in memory
for the duration of the processing of a single document. The persistent entity
never contains the PII value.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

# Closed ENUM with all possible detection systems
class DetectorKind(str, Enum):
    """Detection technique that produced a candidate.

    Closed *architectural* vocabulary (three techniques), not to be confused
    with the PII categories: ``pii_type`` is a string declared in config and
    freely extensible (§2.3.10), whereas the set of techniques is fixed by the
    architecture and adding one is a developer-level activity.

    :cvar REGEX: regular-expression detector, config-driven.
    :cvar NER: zero-shot NER detector (GLiNER).
    :cvar AI: selective generative-AI detector (sampled pass).
    """

    REGEX = "regex"
    NER = "ner"
    AI = "ai"

# "How many of the same opinion exist for the same PII?"
class ConfirmationLevel(str, Enum):
    """Outcome of the merge for a span. Closed architectural vocabulary.

    :cvar SINGLE_SOURCE: detected by a single source, without overlap; never
        discarded (recall-first, §2.5.2).
    :cvar DOUBLE_CONFIRMED: same ``pii_type`` confirmed by regex and NER on
        overlapping spans.
    :cvar CONFLICTING: overlapping spans but disagreeing ``pii_type``; no
        automatic arbitration, resolution is deferred to B5.
    :cvar AI_DISCOVERED: found by the sampled AI pass, absent from the other
        sources.
    """

    SINGLE_SOURCE = "single_source"
    DOUBLE_CONFIRMED = "double_confirmed"
    CONFLICTING = "conflicting"
    AI_DISCOVERED = "ai_discovered"

# String that rapresents a portion of the normalized text
@dataclass(frozen=True)
class TextSpan:
    """Character interval in the normalized text of the document.

    Immutable. Offsets are half-open ``[start, end)`` over the normalized text
    produced by B3.

    :ivar start: starting character offset, inclusive (``>= 0``).
    :ivar end: ending character offset, exclusive (``> start``).
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate the span invariant.

        :raises ValueError: if ``start`` is negative, or if the span is empty or
            inverted (``end <= start``).
        """
        if self.start < 0:
            raise ValueError(f"negative start: {self.start}")
        if self.end <= self.start:
            raise ValueError(f"empty or inverted span: [{self.start}, {self.end})")

    def __len__(self) -> int:
        """:returns: number of characters covered by the span."""
        return self.end - self.start

    def overlaps(self, other: TextSpan) -> bool:
        """Tell whether the two spans share at least one character.

        Two adjacent spans (``self.end == other.start``) do NOT overlap.

        :param other: span to compare against.
        :returns: ``True`` if the intersection is non-empty.
        """
        return self.start < other.end and other.start < self.end

    def overlap_ratio(self, other: TextSpan) -> float:
        """Intersection over Union (IoU) between the two spans.

        Metric used by the :class:`~pii_detection.detection.pipeline.MergeEngine`
        (Step 7) to decide whether two candidates refer to the same match.

        :param other: span to compare against.
        :returns: ratio in ``[0.0, 1.0]``; ``0.0`` if disjoint, ``1.0`` if they
            coincide.
        """
        inter = max(0, min(self.end, other.end) - max(self.start, other.start))
        if inter == 0:
            return 0.0
        union = max(self.end, other.end) - min(self.start, other.start)
        return inter / union

# Where a cerain PII lays in the FS
@dataclass(frozen=True)
class DocumentLocation:
    """Human-facing position of the PII, from the position map provided by B3.

    All fields are optional because they depend on the source format: a PDF has
    a ``page``, a spreadsheet has a ``cell``, plain text may have none.

    :ivar page: page number (1-based), if applicable.
    :ivar paragraph: paragraph index, if applicable.
    :ivar line: line number, if applicable.
    :ivar cell: cell reference (e.g. ``"B4"``) for tabular formats.
    """

    page: int | None = None
    paragraph: int | None = None
    line: int | None = None
    cell: str | None = None


@dataclass(frozen=True)
class NormalizedDocument:
    """Input of the B4 layer: normalized text plus the document identifier.

    The ``TextSpan -> DocumentLocation`` mapping is B3's responsibility (out of
    scope here): until it is available, :meth:`location_for` returns ``None`` and
    the pipeline uses the received value without assuming its presence.

    :ivar document_id: stable identifier of the source document.
    :ivar text: normalized text the detectors operate on.
    """

    document_id: str
    text: str

    def location_for(self, span: TextSpan) -> DocumentLocation | None:
        """Resolve the human-facing position of a span.

        Placeholder awaiting B3: the position map is provided by the extraction
        layer, not by this one.

        :param span: character interval to locate.
        :returns: the corresponding :class:`DocumentLocation`, or ``None`` until
            B3 provides the map.
        """
        return None

# For traceability, can tell which detector found which PII
@dataclass(frozen=True)
class DetectionProvenance:
    """Provenance of a single detection — answers traceability (§2.7.3).

    Immutable. The optional fields are technique-specific: ``raw_label`` for
    NER, ``rationale`` for AI. Keeping them in a single DTO lets
    :class:`PIIMatch` retain heterogeneous provenances in one ``sources`` list.

    :ivar detector_id: id of the detector instance, e.g. ``"regex.iban_v1"``.
    :ivar detector_kind: technique that produced the detection.
    :ivar pii_type: PII category declared in config, e.g. ``"iban"``.
    :ivar confidence: detection confidence in ``[0.0, 1.0]``.
    :ivar raw_label: NER only — textual label passed to the model.
    :ivar rationale: AI only — textual rationale produced by the model.
    """

    detector_id: str
    detector_kind: DetectorKind
    pii_type: str
    confidence: float
    raw_label: str | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        """Validate the confidence range.

        :raises ValueError: if ``confidence`` is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range [0,1]: {self.confidence}")

# Raw output of ONE detector, PRE MERGE
@dataclass
class PIICandidate:
    """Output of ONE detector, pre-merge.

    Mutable for the detectors' convenience during construction. The ``text``
    field lives only in-memory (§2.3.11) and must never reach persistence.

    :ivar span: position of the candidate in the normalized text.
    :ivar text: substring actually detected (in-memory only).
    :ivar provenance: which detector it comes from and with what confidence.
    """

    span: TextSpan
    text: str
    provenance: DetectionProvenance

# Merge OUTPUT, with multiple provenance if possible
@dataclass
class PIIMatch:
    """Unified PII produced by the merge — output of the B4 layer toward B5.

    Aggregates one or more :class:`PIICandidate` insisting on the same span. The
    ``sources`` list (not a single id) retains all the provenances: it is what
    feeds :attr:`ConfirmationLevel.DOUBLE_CONFIRMED` and what the DPO can inspect
    (§2.7.3).

    :ivar span: position of the PII in the normalized text.
    :ivar text: detected value (in-memory only, §2.3.11).
    :ivar pii_type: PII category resulting from the merge.
    :ivar confidence: aggregated confidence in ``[0.0, 1.0]``.
    :ivar confirmation_level: outcome of the merge for this span.
    :ivar sources: provenances contributing to the match (at least one).
    :ivar document_id: document the PII belongs to.
    :ivar location: human-facing position from B3, or ``None`` if unavailable.
    :ivar match_id: detection-time identifier (uuid4); not a persistent key.
    """

    span: TextSpan
    text: str
    pii_type: str
    confidence: float
    confirmation_level: ConfirmationLevel
    sources: list[DetectionProvenance]
    document_id: str
    location: DocumentLocation | None = None
    match_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        """Validate confidence and the presence of at least one provenance.

        :raises ValueError: if ``confidence`` is outside ``[0.0, 1.0]`` or if
            ``sources`` is empty (it would violate traceability, §2.7.3).
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range [0,1]: {self.confidence}")
        if not self.sources:
            raise ValueError("a PIIMatch must have at least one provenance (§2.7.3)")


__all__ = [
    "DetectorKind",
    "ConfirmationLevel",
    "TextSpan",
    "DocumentLocation",
    "NormalizedDocument",
    "DetectionProvenance",
    "PIICandidate",
    "PIIMatch",
]
