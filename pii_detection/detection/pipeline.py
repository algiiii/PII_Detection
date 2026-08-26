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
        ``SINGLE_SOURCE``. The AI candidates are absorbed last, against the matches
        built so far — as a **confirmer** when they agree on ``pii_type`` and a
        **discoverer** when they do not overlap any of them (see
        :meth:`_absorb_ai`).

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

        self._absorb_ai(matches, ai_candidates, document_id)
        return matches

    def _absorb_ai(
        self,
        matches: list[PIIMatch],
        ai_candidates: Sequence[PIICandidate],
        document_id: str,
    ) -> None:
        """Fold the AI candidates into the matches, as confirmer or discoverer.

        Each AI candidate is paired with its best-overlapping match (same IoU
        threshold as the regex×NER pairing), among the matches that existed
        **before** AI absorption — a newly added ``AI_DISCOVERED`` match is never a
        pairing target, so the AI never confirms itself. The outcome, per the
        design table:

        =========================================  =====================================
        AI candidate vs its best match             Outcome
        =========================================  =====================================
        no match with IoU ≥ threshold              new ``AI_DISCOVERED`` match
        best match, **same** ``pii_type``          confirm in place (see :meth:`_confirm_with_ai`)
        best match, **different** ``pii_type``     AI kept as its own ``CONFLICTING``
                                                   match; the existing one is demoted to
                                                   ``CONFLICTING`` only if it was
                                                   ``SINGLE_SOURCE`` (a regex+NER
                                                   agreement is not undone by a single
                                                   AI opinion)
        =========================================  =====================================

        Mutates ``matches`` in place (recall-first: nothing is discarded).

        :param matches: the matches built from regex/NER, extended in place.
        :param ai_candidates: candidates of the sampled AI pass.
        :param document_id: document the resulting matches belong to.
        """
        pre_ai = list(matches)
        for ai in ai_candidates:
            best = self._best_overlap(ai, pre_ai)
            if best is None:
                matches.append(
                    self._single(ai, document_id, level=ConfirmationLevel.AI_DISCOVERED)
                )
            elif best.pii_type == ai.provenance.pii_type:
                self._confirm_with_ai(best, ai)
            else:
                if best.confirmation_level is ConfirmationLevel.SINGLE_SOURCE:
                    best.confirmation_level = ConfirmationLevel.CONFLICTING
                matches.append(
                    self._single(ai, document_id, level=ConfirmationLevel.CONFLICTING)
                )

    def _best_overlap(
        self, candidate: PIICandidate, matches: Sequence[PIIMatch]
    ) -> PIIMatch | None:
        """Return the match with the highest IoU over the threshold, or ``None``.

        :param candidate: the candidate looking for a partner match.
        :param matches: the matches to search.
        :returns: the best-overlapping match with IoU ≥ ``min_overlap_ratio``, or
            ``None`` if none reaches the threshold.
        """
        best, best_iou = None, -1.0
        for match in matches:
            iou = candidate.span.overlap_ratio(match.span)
            if iou >= self.min_overlap_ratio and iou > best_iou:
                best, best_iou = match, iou
        return best

    def _confirm_with_ai(self, match: PIIMatch, ai: PIICandidate) -> None:
        """Append the AI provenance to a match agreeing on ``pii_type``.

        The AI provenance is added to ``sources`` and the confidence gains
        :attr:`double_confirmation_bonus` (clamped to ``1.0``). The level escalates
        ``SINGLE_SOURCE → DOUBLE_CONFIRMED``; a ``DOUBLE_CONFIRMED`` match stays so
        (now carrying three provenances) and a ``CONFLICTING`` one stays
        ``CONFLICTING`` (its type disagreement is arbitrated by B5, not resolved by
        an AI agreement). An AI detector already present in ``sources`` is a no-op,
        so a value re-seen across chunk overlaps cannot inflate the confidence.

        :param match: the match to confirm, mutated in place.
        :param ai: the agreeing AI candidate.
        """
        if any(s.detector_id == ai.provenance.detector_id for s in match.sources):
            return
        match.sources.append(ai.provenance)
        match.confidence = min(1.0, match.confidence + self.double_confirmation_bonus)
        if match.confirmation_level is ConfirmationLevel.SINGLE_SOURCE:
            match.confirmation_level = ConfirmationLevel.DOUBLE_CONFIRMED

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
