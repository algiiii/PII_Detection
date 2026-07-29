"""Tests for the corpus renderer (Tier 2 input).

Rendering needs ``fpdf2`` (and ``python-docx`` for DOCX); the round-trip also
needs ``fitz``. Each test ``importorskip``s what it uses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("fpdf")

from pii_detection.evaluation.render import render_corpus  # noqa: E402


def _norm(text: str) -> str:
    """Collapse whitespace, so a value split across wrapped lines still matches."""
    return re.sub(r"\s+", " ", text)


def test_render_writes_pdfs_and_gold(tmp_path: Path) -> None:
    gold = render_corpus(tmp_path, n=4, seed=1, formats=["pdf"])
    assert len(sorted(tmp_path.glob("*.pdf"))) == 4

    records = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    first = records[0]
    assert first["document_id"] == "gen_0001"
    assert first["pii"]
    assert all({"pii_type", "value"} <= item.keys() for item in first["pii"])


def test_render_docx(tmp_path: Path) -> None:
    pytest.importorskip("docx")
    render_corpus(tmp_path, n=2, seed=1, formats=["docx"])
    assert len(sorted(tmp_path.glob("*.docx"))) == 2


def test_rendered_pdf_extracts_back_every_gold_value(tmp_path: Path) -> None:
    """End-to-end sanity: render -> extract recovers every injected value (after
    whitespace normalization), so the renderer is not the reason a value is lost."""
    pytest.importorskip("fitz")
    from pii_detection.extraction import extract_document

    gold = render_corpus(tmp_path, n=6, seed=42, formats=["pdf"])
    for line in gold.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        text = _norm(extract_document(tmp_path / f"{record['document_id']}.pdf").text)
        for item in record["pii"]:
            assert _norm(item["value"]) in text
