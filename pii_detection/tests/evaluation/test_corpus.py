"""Tests for the annotated evaluation corpus loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from pii_detection.evaluation.corpus import (
    AnnotatedDocument,
    load_annotated_corpus,
    load_corpus_dir,
    load_corpus_jsonl,
    parse_annotated_text,
    sample_documents,
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


class TestLoadJsonl:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_corpus_jsonl(tmp_path / "assente.jsonl")

    def test_parses_records_in_order(self, tmp_path: Path) -> None:
        path = tmp_path / "sources.jsonl"
        path.write_text(
            '{"document_id": "a/b.pdf", "annotated": "x {{email:a@b.c}}"}\n'
            "\n"  # blank lines are skipped
            '{"document_id": "c.txt", "annotated": "niente"}\n',
            encoding="utf-8",
        )
        docs = load_corpus_jsonl(path)
        assert [d.document_id for d in docs] == ["a/b.pdf", "c.txt"]
        assert docs[0].spans[0].pii_type == "email"
        assert docs[1].spans == ()

    def test_malformed_line_raises_with_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "sources.jsonl"
        path.write_text('{"document_id": "a", "annotated": "ok"}\n{"nope": 1}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2:"):
            load_corpus_jsonl(path)


class TestLoadAnnotatedCorpus:
    def test_dispatches_on_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x {{iban:IT60X05}}", encoding="utf-8")
        assert [d.document_id for d in load_annotated_corpus(tmp_path)] == ["a"]

    def test_dispatches_on_jsonl_file(self, tmp_path: Path) -> None:
        path = tmp_path / "sources.jsonl"
        path.write_text('{"document_id": "d.pdf", "annotated": "{{phone:+39 0}}"}\n', encoding="utf-8")
        assert [d.document_id for d in load_annotated_corpus(path)] == ["d.pdf"]

    def test_none_falls_back_to_packaged_corpus(self) -> None:
        assert len(load_annotated_corpus(None)) >= 1

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_annotated_corpus(tmp_path / "assente")


class TestSampleDocuments:
    @staticmethod
    def _corpus(n: int) -> list[AnnotatedDocument]:
        return [
            parse_annotated_text(f"doc{i:03}", f"x {{{{email:a{i}@b.c}}}}") for i in range(n)
        ]

    def test_draw_is_reproducible_for_a_given_seed(self) -> None:
        corpus = self._corpus(50)
        first = [d.document_id for d in sample_documents(corpus, 10, seed=7)]
        second = [d.document_id for d in sample_documents(corpus, 10, seed=7)]
        assert first == second
        assert len(first) == 10

    def test_different_seeds_draw_differently(self) -> None:
        corpus = self._corpus(50)
        assert [d.document_id for d in sample_documents(corpus, 10, seed=1)] != [
            d.document_id for d in sample_documents(corpus, 10, seed=2)
        ]

    def test_draw_keeps_corpus_order(self) -> None:
        corpus = self._corpus(50)
        drawn = [d.document_id for d in sample_documents(corpus, 10, seed=3)]
        assert drawn == sorted(drawn)

    def test_size_beyond_corpus_returns_everything(self) -> None:
        corpus = self._corpus(5)
        assert len(sample_documents(corpus, 99)) == 5

    def test_non_positive_size_raises(self) -> None:
        with pytest.raises(ValueError):
            sample_documents(self._corpus(5), 0)
