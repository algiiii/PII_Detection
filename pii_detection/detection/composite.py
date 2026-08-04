"""Combine several detectors into one — run multiple sources in a single slot.

The pipeline hands the merge a single "pattern" detector
(``merge.merge(pattern.detect(text), ner.detect(text))``). :class:`CompositeDetector`
lets that slot be more than one source — for example Presidio's built-in pattern
recognizers **plus** a config-driven
:class:`~pii_detection.detection.regex_detector.RegexDetector` for custom patterns
— without changing the merge or the entry points. Each wrapped detector keeps its
own provenance on the candidates it produces; the composite only concatenates the
lists.
"""

from __future__ import annotations

from collections.abc import Sequence

from pii_detection.detection.protocol import BaseDetector, PIIDetector
from pii_detection.detection.types import DetectorKind, PIICandidate


class CompositeDetector(BaseDetector):
    """A detector that runs several detectors and concatenates their candidates.

    :ivar detector_id: identifier of the composite instance.
    :ivar detector_kind: technique tag of the composite (default ``REGEX``).
    """

    def __init__(
        self,
        detector_id: str,
        detectors: Sequence[PIIDetector],
        *,
        detector_kind: DetectorKind = DetectorKind.REGEX,
    ) -> None:
        """Store the wrapped detectors.

        :param detector_id: identifier of the composite instance.
        :param detectors: the detectors to run, in order.
        :param detector_kind: technique tag for the composite; defaults to ``REGEX``.
        """
        super().__init__(detector_id, detector_kind)
        self._detectors = list(detectors)

    def detect(self, text: str) -> list[PIICandidate]:
        """Run every wrapped detector and concatenate the candidates.

        :param text: normalized document text to scan.
        :returns: all candidates from all wrapped detectors, in detector order;
            possibly empty, never ``None``.
        """
        candidates: list[PIICandidate] = []
        for detector in self._detectors:
            candidates.extend(detector.detect(text))
        return candidates


__all__ = ["CompositeDetector"]
