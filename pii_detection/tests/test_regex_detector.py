"""Tests for the config-driven regex detector (Step 4)."""

from __future__ import annotations

import pytest

from pii_detection.detection.config import RegexRuleModel
from pii_detection.detection.protocol import PIIDetector
from pii_detection.detection.regex_detector import RegexDetector
from pii_detection.detection.types import DetectorKind


def _rule(
    rule_id: str,
    pii_type: str,
    pattern: str,
    *,
    base_confidence: float = 0.6,
    flags: tuple[str, ...] = ("UNICODE",),
) -> RegexRuleModel:
    """Build a rule in memory, bypassing YAML (the detector takes rules directly)."""
    return RegexRuleModel(
        rule_id=rule_id,
        pii_type=pii_type,
        pattern=pattern,
        base_confidence=base_confidence,
        flags=flags,
    )


def _detector(*rules: RegexRuleModel) -> RegexDetector:
    """Build a detector from the given rules under a fixed id."""
    return RegexDetector("regex.test", list(rules))


class TestMatching:
    def test_two_rules_are_both_applied(self) -> None:
        """Every rule must run. This is the guard against an early ``return``
        inside the rules loop: with the bug, only the first rule would fire."""
        det = _detector(
            _rule("email", "email", r"\b\w+@\w+\.\w+\b"),
            _rule("iban", "iban", r"\bIT\d{2}[A-Z0-9]+\b"),
        )
        text = "scrivi a bob@acme.com oppure IT60X0542811101000000123456 grazie"
        found = {c.provenance.pii_type for c in det.detect(text)}
        assert found == {"email", "iban"}

    def test_all_occurrences_of_one_rule(self) -> None:
        """``finditer`` must report every non-overlapping match, not just the first."""
        det = _detector(_rule("email", "email", r"\b\w+@\w+\.\w+\b"))
        cands = det.detect("a@b.com and c@d.net")
        assert [c.text for c in cands] == ["a@b.com", "c@d.net"]

    def test_no_match_returns_empty_list(self) -> None:
        """No PII in the text -> empty list, never ``None``."""
        det = _detector(_rule("iban", "iban", r"\bIT\d{2}\b"))
        assert det.detect("nessun dato qui") == []


class TestOffsets:
    @pytest.mark.parametrize(
        ("text", "start", "end", "matched"),
        [
            ("aXXb", 1, 3, "XX"),
            ("XX", 0, 2, "XX"),
            ("  XX  ", 2, 4, "XX"),
        ],
    )
    def test_span_and_text_align(
        self, text: str, start: int, end: int, matched: str
    ) -> None:
        """The span offsets must point at the exact substring, and the candidate's
        ``text`` must equal ``text[start:end]`` (the slice done by build_candidate)."""
        det = _detector(_rule("x", "token", "XX"))
        (cand,) = det.detect(text)  # exactly one match expected
        assert cand.span.start == start
        assert cand.span.end == end
        assert cand.text == matched
        assert text[cand.span.start : cand.span.end] == cand.text


class TestZeroWidthGuard:
    def test_empty_matches_are_skipped(self) -> None:
        """A pattern that can match the empty string (``x*`` on a text with no
        'x') yields zero-width matches: they must be skipped, not crash the
        ``TextSpan`` invariant ``end > start``."""
        det = _detector(_rule("z", "noise", "x*"))
        assert det.detect("abc") == []


class TestProvenance:
    def test_stamps_identity_and_rule_data(self) -> None:
        """Each candidate carries the detector identity and the rule's data:
        ``detector_kind`` is always REGEX, ``pii_type`` and ``confidence`` come
        from the rule (traceability, §2.7.3)."""
        det = _detector(_rule("iban", "iban", r"\bIT\d{2}\b", base_confidence=0.6))
        (cand,) = det.detect("IT60")
        prov = cand.provenance
        assert prov.detector_id == "regex.test"
        assert prov.detector_kind is DetectorKind.REGEX
        assert prov.pii_type == "iban"
        assert prov.confidence == pytest.approx(0.6)


class TestFlags:
    def test_ignorecase_flag_is_honored(self) -> None:
        """A rule declaring IGNORECASE matches case-insensitively; the default
        (UNICODE only) does not — proving ``re_flags`` reaches ``re.compile``."""
        assert len(_detector(_rule("c", "kw", "abc", flags=("IGNORECASE",))).detect("ABC")) == 1
        assert _detector(_rule("c", "kw", "abc")).detect("ABC") == []


class TestContract:
    def test_satisfies_pii_detector_protocol(self) -> None:
        """Structural typing: RegexDetector is a PIIDetector without inheriting
        the Protocol, thanks to ``@runtime_checkable`` + the ``detect`` method."""
        assert isinstance(_detector(_rule("x", "t", "a")), PIIDetector)


def _highlight(text: str, detector: RegexDetector) -> str:
    """Return ``text`` with every match wrapped in ANSI reverse-video for the terminal.

    Substitutions are applied back-to-front (matches sorted by descending
    start offset) so inserting the escape codes never shifts the offsets of
    the matches still to be wrapped.

    :param text: text to scan and render.
    :param detector: detector whose matches are highlighted.
    :returns: the same text with ANSI escapes around each matched span.
    """
    out = text
    for cand in sorted(detector.detect(text), key=lambda c: c.span.start, reverse=True):
        start, end = cand.span.start, cand.span.end
        out = f"{out[:start]}\033[7m{out[start:end]}\033[0m{out[end:]}"
    return out


class TestVisual:
    @pytest.mark.manual
    def test_show_matches_highlighted(self) -> None:
        """Manual eyeball check: prints the matches highlighted in the terminal.

        Not part of the automated suite (marker ``manual`` is deselected by
        default). Run it with the matched spans on screen via::

            ./.venv/bin/pytest -m manual -s -k show_matches_highlighted
        """
        det = _detector(
            _rule("email", "email", r"\b\w+@\w+\.\w+\b"),
            _rule("iban", "iban", r"\bCH\d{2}(?:[ ]?\d){15,}\b"),
        )
        text = "Scrivimi a mario@acme.com, IBAN CH93 0076 2011 6238 5295 7, grazie."
        print("\n" + _highlight(text, det))
