"""Merge of the detector candidates into unified matches (block B4).

Turns the raw :class:`~pii_detection.detection.types.PIICandidate` produced by
each source (regex/pattern, NER, sampled AI) into unified
:class:`~pii_detection.detection.types.PIIMatch`. There is no intermediate data
model: the certainty of a detection is carried by ``PIIMatch`` itself —
``confidence`` (strength), ``confirmation_level`` (how it was confirmed) and
``sources`` (which and how many detectors agree). The :class:`MergeEngine` only
decides those three fields per span; binding to a document is a caller concern
(the ``document_id`` is passed in), which keeps the merge testable with
synthetic candidates.
"""

from __future__ import annotations

from collections.abc import Sequence

from pii_detection.detection.types import (
    ConfirmationLevel,
    PIICandidate,
    PIIMatch,
)


class MergeEngine:
    """Combine the per-source candidates into unified matches.

    Cross-matches the regex and NER sources by span overlap (Intersection over
    Union) and absorbs the sampled AI candidates last, producing
    :class:`~pii_detection.detection.types.PIIMatch` objects. Consistent with the
    recall-first principle (§2.5.2), no candidate is ever discarded: a weak or
    lone one is kept and flagged, never dropped.

    :ivar min_overlap_ratio: minimum span overlap (IoU, see
        :meth:`~pii_detection.detection.types.TextSpan.overlap_ratio`) for two
        candidates to count as the same PII.
    :ivar double_confirmation_bonus: confidence added when two sources agree on
        the same ``pii_type`` over overlapping spans (result clamped to ``1.0``).
    """

    def __init__(
        self,
        *,
        min_overlap_ratio: float = 0.5,
        double_confirmation_bonus: float = 0.15,
    ) -> None:
        """Store the merge tunables.

        :param min_overlap_ratio: overlap threshold in ``[0, 1]``.
        :param double_confirmation_bonus: bonus in ``[0, 1]``.
        :raises ValueError: if either value falls outside ``[0, 1]``.
        """
        if not 0.0 <= min_overlap_ratio <= 1.0:
            raise ValueError(f"min_overlap_ratio out of [0,1]: {min_overlap_ratio}")
        if not 0.0 <= double_confirmation_bonus <= 1.0:
            raise ValueError(
                f"double_confirmation_bonus out of [0,1]: {double_confirmation_bonus}"
            )
        self.min_overlap_ratio = min_overlap_ratio
        self.double_confirmation_bonus = double_confirmation_bonus

    def merge(
        self,
        regex_candidates: Sequence[PIICandidate],
        ner_candidates: Sequence[PIICandidate],
        ai_candidates: Sequence[PIICandidate] = (),
        *,
        document_id: str,
    ) -> list[PIIMatch]:
        """Unify the candidates of the sources into matches.

        Each regex candidate is paired with its best-overlapping NER candidate:
        same ``pii_type`` yields a ``DOUBLE_CONFIRMED`` match, a disagreeing type
        keeps both as ``CONFLICTING`` (arbitration deferred to B5), no partner
        yields a ``SINGLE_SOURCE`` match. NER candidates left unpaired are kept as
        ``SINGLE_SOURCE``. The AI candidates are absorbed last, only where they
        cover a span no other source already touches (``AI_DISCOVERED``).

        :param regex_candidates: candidates of the regex/pattern sources.
        :param ner_candidates: candidates of the NER source.
        :param ai_candidates: candidates of the sampled AI pass (may be empty).
        :param document_id: document the resulting matches belong to.
        :returns: the unified matches, in no particular order.
        """
        matches: list[PIIMatch] = []
        used_ner: set[int] = set()

        for regex in regex_candidates:
            best_j, best_iou = -1, -1.0
            for j, ner in enumerate(ner_candidates):
                if j in used_ner:
                    continue
                iou = regex.span.overlap_ratio(ner.span)
                if iou >= self.min_overlap_ratio and iou > best_iou:
                    best_j, best_iou = j, iou
            if best_j < 0:  # no overlapping NER candidate
                matches.append(self._single(regex, document_id))
                continue
            ner = ner_candidates[best_j]
            used_ner.add(best_j)
            if regex.provenance.pii_type == ner.provenance.pii_type:
                matches.append(self._double(regex, ner, document_id))
            else:  # overlap but disagreeing type: keep both, B5 arbitrates
                matches.append(
                    self._single(regex, document_id, level=ConfirmationLevel.CONFLICTING)
                )
                matches.append(
                    self._single(ner, document_id, level=ConfirmationLevel.CONFLICTING)
                )

        for j, ner in enumerate(ner_candidates):
            if j not in used_ner:
                matches.append(self._single(ner, document_id))

        for ai in ai_candidates:
            if any(ai.span.overlaps(match.span) for match in matches):
                continue
            matches.append(
                self._single(ai, document_id, level=ConfirmationLevel.AI_DISCOVERED)
            )

        return matches

    def _single(
        self,
        candidate: PIICandidate,
        document_id: str,
        *,
        level: ConfirmationLevel = ConfirmationLevel.SINGLE_SOURCE,
    ) -> PIIMatch:
        """Build a single-provenance match from a lone candidate.

        :param candidate: the candidate to promote to a match.
        :param document_id: document the match belongs to.
        :param level: confirmation level to stamp; ``SINGLE_SOURCE`` by default,
            ``CONFLICTING`` or ``AI_DISCOVERED`` for the respective cases.
        :returns: the built match, carrying the candidate's own provenance.
        """
        provenance = candidate.provenance
        return PIIMatch(
            span=candidate.span,
            text=candidate.text,
            pii_type=provenance.pii_type,
            confidence=provenance.confidence,
            confirmation_level=level,
            sources=[provenance],
            document_id=document_id,
        )

    def _double(
        self, regex: PIICandidate, ner: PIICandidate, document_id: str
    ) -> PIIMatch:
        """Build a double-confirmed match from two agreeing candidates.

        The regex span and text are kept as representative (pattern spans are
        exact, NER ones can be loose), the confidence is the stronger of the two
        plus :attr:`double_confirmation_bonus`, clamped to ``1.0``, and both
        provenances are retained in ``sources``.

        :param regex: the regex/pattern candidate.
        :param ner: the NER candidate agreeing on ``pii_type``.
        :param document_id: document the match belongs to.
        :returns: the double-confirmed match.
        """
        confidence = min(
            1.0,
            max(regex.provenance.confidence, ner.provenance.confidence)
            + self.double_confirmation_bonus,
        )
        return PIIMatch(
            span=regex.span,
            text=regex.text,
            pii_type=regex.provenance.pii_type,
            confidence=confidence,
            confirmation_level=ConfirmationLevel.DOUBLE_CONFIRMED,
            sources=[regex.provenance, ner.provenance],
            document_id=document_id,
        )


__all__ = ["MergeEngine"]
