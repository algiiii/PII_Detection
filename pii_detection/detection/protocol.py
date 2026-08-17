"""Detector contract and optional base for code reuse.

Defined in ``doc/planning.md`` §"Contratto detector: Protocol, non ABC".

The orchestrator :class:`~pii_detection.detection.pipeline.HybridDetectionPipeline`
depends only on the *shape* of a detector — ``detect(text) -> list[PIICandidate]``
— not on an inheritance hierarchy. That is why the contract is a
:class:`typing.Protocol` (structural typing, PEP 544) instead of an abstract
base class: a fake in the tests or a third-party wrapper that already exposes
``detect()`` satisfies the contract without inheriting anything.

:class:`BaseDetector`, by contrast, is an *optional* concrete class: it imposes
nothing, it only offers shared utilities (storing id/kind and building a
:class:`~pii_detection.detection.types.PIICandidate`) to the three real
detectors.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pii_detection.detection.types import (
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    TextSpan,
)


@runtime_checkable
class PIIDetector(Protocol):
    """Shape every detector must expose to be used by the pipeline.

    Minimal contract, verifiable at runtime with ``isinstance`` (only the
    presence of the :meth:`detect` method; the attributes are guaranteed by type
    checking).

    :ivar detector_id: identifier of the instance, e.g. ``"regex.iban_v1"``.
    :ivar detector_kind: technique implemented by the detector.
    """

    detector_id: str
    detector_kind: DetectorKind

    def detect(self, text: str) -> list[PIICandidate]:
        """Search for PII in the text and return the raw candidates.

        Consistent with the recall-first principle (§2.5.2): a detector never
        discards a weak candidate on its own, it flags it with low confidence.

        :param text: normalized document text to operate on.
        :returns: candidates found, possibly empty; never ``None``.
        """
        ...


class BaseDetector:
    """*Optional* concrete base with utilities common to the real detectors.

    It is not abstract and does not implement :meth:`~PIIDetector.detect`: the
    subclasses (``RegexDetector``, ``PresidioDetector``, ``LLMDetector``) do.
    Inheriting it is a convenience, not a contract requirement.

    :ivar detector_id: identifier of the instance.
    :ivar detector_kind: technique implemented.
    """

    def __init__(self, detector_id: str, detector_kind: DetectorKind) -> None:
        """Store the detector's identity.

        :param detector_id: unique identifier of the instance.
        :param detector_kind: technique implemented by the detector.
        """
        self.detector_id = detector_id
        self.detector_kind = detector_kind

    def build_candidate(
        self,
        text: str,
        span: TextSpan,
        pii_type: str,
        confidence: float,
        *,
        raw_label: str | None = None,
        rationale: str | None = None,
    ) -> PIICandidate:
        """Build a candidate stamping it with the detector's provenance.

        Extracts the substring ``text[span.start:span.end]`` and wraps it in a
        :class:`~pii_detection.detection.types.PIICandidate`, sparing every
        detector from repeating the text slicing and the population of
        :class:`~pii_detection.detection.types.DetectionProvenance` with its own
        ``detector_id`` and ``detector_kind``.

        :param text: full document text (the span is extracted from it).
        :param span: position of the candidate in the text.
        :param pii_type: PII category declared in config, e.g. ``"iban"``.
        :param confidence: detection confidence in ``[0.0, 1.0]``.
        :param raw_label: NER only — textual label passed to the model.
        :param rationale: AI only — textual rationale from the model.
        :returns: the candidate ready for the merge phase.
        :raises ValueError: if ``span`` exceeds the length of ``text`` or if
            ``confidence`` is outside ``[0.0, 1.0]`` (validated by
            :class:`~pii_detection.detection.types.DetectionProvenance`).
        """
        if span.end > len(text):
            raise ValueError(f"span {span} beyond end of text (len={len(text)})")
        provenance = DetectionProvenance(
            detector_id=self.detector_id,
            detector_kind=self.detector_kind,
            pii_type=pii_type,
            confidence=confidence,
            raw_label=raw_label,
            rationale=rationale,
        )
        return PIICandidate(
            span=span,
            text=text[span.start : span.end],
            provenance=provenance,
        )


__all__ = ["PIIDetector", "BaseDetector"]
