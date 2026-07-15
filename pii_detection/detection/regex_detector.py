from __future__ import annotations

import re

from pii_detection.detection.protocol import BaseDetector
from pii_detection.detection.types import TextSpan, PIICandidate, DetectorKind
from pii_detection.detection.config import RegexRuleModel


class RegexDetector(BaseDetector):
    """Config-driven regex detector (block B4, Step 4).

    A single generic detector instantiated from a list of
    :class:`~pii_detection.detection.config.RegexRuleModel`: adding a
    regex-based PII category is a YAML edit, not a new Python subclass
    (§2.3.10). Each rule's pattern is compiled once, at construction.

    Consistent with **recall-first (§2.5.2)**, it never discards a match: every
    occurrence of every rule becomes a
    :class:`~pii_detection.detection.types.PIICandidate` at the rule's
    ``base_confidence``.
    """

    def __init__(self, detector_id: str, rules: list[RegexRuleModel]) -> None:
        """Store the detector identity and pre-compile the rules.

        :param detector_id: identifier of the instance, e.g. ``"regex.main"``.
        :param rules: validated regex rules the detector runs; each pattern is
            compiled with its ``re_flags`` and kept paired with the rule it
            came from.
        """
        # Call super to initialize the detector
        super().__init__(detector_id, DetectorKind.REGEX)
        # Compile each rule, keeping it next to its compiled pattern
        self._compiled = [(re.compile(r.pattern, r.re_flags), r) for r in rules]

    def detect(self, text: str) -> list[PIICandidate]:
        """Find every regex match in the text and wrap it in a candidate.

        Each rule is applied with ``finditer``, so all non-overlapping
        occurrences are reported, not just the first. Zero-width matches are
        skipped, because an empty span is invalid for
        :class:`~pii_detection.detection.types.TextSpan`.

        :param text: normalized document text to scan.
        :returns: one candidate per match, possibly empty; never ``None``. No
            match is discarded (recall-first, §2.5.2).
        """
        candidates: list[PIICandidate] = []
        for pattern, rule in self._compiled:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if start == end:
                    continue
                span = TextSpan(start, end)
                candidates.append(
                    self.build_candidate(text, span, rule.pii_type, rule.base_confidence)
                )
        return candidates


__all__ = ["RegexDetector"]
