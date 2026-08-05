"""Tests for the minimal B3 extraction layer.

Plain-text extraction, dispatch and error paths need no optional deps. The
PDF/DOCX/XLSX round-trips build a tiny file and read it back, so they
``importorskip`` the ``[extraction]``/``[eval]`` libraries and are skipped when
absent. Legacy ``.doc`` is covered only on its guard path (antiword absent), since
a valid ``.doc`` is impractical to author in a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pii_detection.extraction.extractor import (
    UnsupportedFormatError,
    extract_document,
    normalize_text,
    supported_suffixes,
)


def test_txt_extraction_sets_id_and_text(tmp_path: Path) -> None:
    path = tmp_path / "doc_a.txt"
    path.write_text("Il dipendente Mario Rossi.\n", encoding="utf-8")
    doc = extract_document(path)
    assert doc.document_id == "doc_a"
    assert "Mario Rossi" in doc.text


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        extract_document(path)


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        extract_document("/definitely/not/here.pdf")


def test_supported_suffixes() -> None:
    assert supported_suffixes() == frozenset(
        {".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xlsm", ".ods"}
    )


def test_normalize_tidies_whitespace_without_moving_tokens() -> None:
    assert normalize_text("a  \n\n\n\nb\r\nc\n") == "a\n\nb\nc"


def test_docx_round_trip(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "hr_01.docx"
    document = docx.Document()
    document.add_paragraph("Codice fiscale RSSMRA85T10A562S del dipendente.")
    document.save(str(path))

    out = extract_document(path)
    assert out.document_id == "hr_01"
    assert "RSSMRA85T10A562S" in out.text


def test_pdf_round_trip(tmp_path: Path) -> None:
    fpdf = pytest.importorskip("fpdf")
    pytest.importorskip("fitz")
    path = tmp_path / "letter_01.pdf"
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "IBAN IT60X0542811101000000123456")
    pdf.output(str(path))

    out = extract_document(path)
    assert out.document_id == "letter_01"
    assert "IT60X0542811101000000123456" in out.text


def test_xlsx_round_trip(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "clienti_01.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Nome", "Email"])
    sheet.append(["Mario Rossi", "mario@x.it"])
    workbook.save(str(path))

    out = extract_document(path)
    assert out.document_id == "clienti_01"
    assert "Mario Rossi" in out.text
    assert "mario@x.it" in out.text


def test_doc_without_antiword_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A valid .doc is hard to author in tests; here we verify the guard: with the
    # antiword binary absent, a .doc is reported as unsupported (the batch then skips
    # it) rather than crashing.
    monkeypatch.setattr("shutil.which", lambda _name: None)
    path = tmp_path / "legacy_01.doc"
    path.write_bytes(b"\xd0\xcf\x11\xe0 fake legacy word")
    with pytest.raises(UnsupportedFormatError):
        extract_document(path)
