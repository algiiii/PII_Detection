"""Tests for the annotated evaluation corpus loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from pii_detection.evaluation.corpus import (
    load_corpus_dir,
    parse_annotated_text,
)


class TestParse:
    def test_markers_are_stripped(self) -> None:
        """The clean text must contain the values but none of the markers."""
        doc = parse_annotated_text("d", "scrivi a {{email:x@y.z}} ora")
        assert doc.text == "scrivi a x@y.z ora"
        assert "{{" not in doc.text and "}}" not in doc.text

    def test_span_points_at_value(self) -> None:
        """The computed span must slice out exactly the annotated value."""
        doc = parse_annotated_text("d", "IBAN {{iban:IT60X0}} ok")
        (span,) = doc.spans
        assert doc.text[span.start : span.end] == "IT60X0"
        assert span.pii_type == "iban"

    def test_multiple_annotations_keep_order_and_offsets(self) -> None:
        """Several markers: types in order, and each span aligned on the value."""
        doc = parse_annotated_text("d", "{{iban:IT00}} poi {{email:a@b.c}}")
        assert [s.pii_type for s in doc.spans] == ["iban", "email"]
        assert [doc.text[s.start : s.end] for s in doc.spans] == ["IT00", "a@b.c"]

    def test_no_annotations(self) -> None:
        """Plain text with no PII yields the text unchanged and no spans."""
        doc = parse_annotated_text("d", "testo senza dati")
        assert doc.text == "testo senza dati"
        assert doc.spans == ()


class TestLoadDir:
    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_corpus_dir(tmp_path / "assente")

    def test_loads_txt_sorted_by_name(self, tmp_path: Path) -> None:
        (tmp_path / "b.txt").write_text("no pii", encoding="utf-8")
        (tmp_path / "a.txt").write_text("x {{email:a@b.c}}", encoding="utf-8")
        docs = load_corpus_dir(tmp_path)
        assert [d.document_id for d in docs] == ["a", "b"]

    def test_shipped_corpus_loads(self) -> None:
        """The documents shipped in evaluation/documents parse without error."""
        docs = load_corpus_dir()
        assert len(docs) >= 1
        assert any(s.pii_type == "iban" for d in docs for s in d.spans)
