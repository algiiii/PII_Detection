"""Tests for the document reference date (B3) the retention check (B7) works on.

Two things matter here and are tested separately: that the *better* evidence wins
when a file carries a usable internal date, and that a **dirty** internal date is
rejected rather than believed. The second is the reason the module exists in the
first place — a metadata field defaulted to the epoch, or set in the future by a
wrong clock, must not silently become a document's age.

The format cases build a tiny real file and read it back, so they ``importorskip``
the ``[extraction]``/``[eval]`` libraries and are skipped when absent.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pii_detection.extraction.dates import (
    DateSource,
    _parse_pdf_date,
    reference_date,
)

_NOW = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
_MTIME = datetime(2024, 5, 4, 9, 30, tzinfo=timezone.utc)


def _with_mtime(path: Path, when: datetime = _MTIME) -> Path:
    """Stamp a known modification time on a file, so the fallback is recognisable."""
    stamp = when.timestamp()
    os.utime(path, (stamp, stamp))
    return path


def _repack_xlsx_modified(source: Path, target: Path, stamp: str) -> Path:
    """Copy an ``.xlsx``, forcing ``dcterms:modified`` in its core properties."""
    import re
    import zipfile

    # Only the text node is replaced: the tag carries its own xmlns declarations,
    # and dropping them yields an XML the reader cannot parse.
    with zipfile.ZipFile(source) as archive, zipfile.ZipFile(
        target, "w", zipfile.ZIP_DEFLATED
    ) as out:
        for item in archive.infolist():
            data = archive.read(item.filename)
            if item.filename == "docProps/core.xml":
                data = re.sub(
                    rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)",
                    rb"\g<1>" + stamp.encode() + rb"\g<2>",
                    data,
                )
            out.writestr(item, data)
    return target


def _pdf_with_metadata(tmp_path: Path, metadata: dict[str, str]) -> Path:
    """Write a one-line PDF carrying the given info-dictionary entries."""
    fpdf = pytest.importorskip("fpdf")
    fitz = pytest.importorskip("fitz")
    raw = tmp_path / "raw.pdf"
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "Contratto")
    pdf.output(str(raw))

    path = tmp_path / "contratto.pdf"
    with fitz.open(raw) as document:
        document.set_metadata(metadata)
        document.save(str(path))
    return _with_mtime(path)


# --- fallback path -----------------------------------------------------------


def test_txt_falls_back_to_mtime(tmp_path: Path) -> None:
    path = tmp_path / "nota.txt"
    path.write_text("x", encoding="utf-8")
    _with_mtime(path)

    result = reference_date(path, now=_NOW)
    assert result.source is DateSource.FILE_MTIME
    assert result.field == "fs:mtime"
    assert result.value == _MTIME


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        reference_date("/definitely/not/here.pdf")


def test_unreadable_metadata_falls_back_instead_of_raising(tmp_path: Path) -> None:
    # A file that claims to be a PDF and is not: reading its metadata blows up, and
    # the fallback must absorb it — refusing to date a document would abort its
    # ingestion for no gain.
    pytest.importorskip("fitz")
    path = tmp_path / "corrotto.pdf"
    path.write_bytes(b"not a pdf at all")
    _with_mtime(path)

    result = reference_date(path, now=_NOW)
    assert result.source is DateSource.FILE_MTIME
    assert result.value == _MTIME


# --- metadata path -----------------------------------------------------------


def test_pdf_modification_date_wins_over_creation_and_mtime(tmp_path: Path) -> None:
    path = _pdf_with_metadata(
        tmp_path,
        {"modDate": "D:20190312101500+01'00'", "creationDate": "D:20210101000000Z"},
    )
    result = reference_date(path, now=_NOW)
    assert result.source is DateSource.CONTENT_METADATA
    assert result.field == "pdf:modDate"
    assert result.value == datetime(2019, 3, 12, 9, 15, tzinfo=timezone.utc)


def test_pdf_creation_date_used_when_modification_absent(tmp_path: Path) -> None:
    path = _pdf_with_metadata(tmp_path, {"modDate": "", "creationDate": "D:20200607"})
    result = reference_date(path, now=_NOW)
    assert result.field == "pdf:creationDate"
    assert result.value == datetime(2020, 6, 7, tzinfo=timezone.utc)


def test_docx_core_properties(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "verbale.docx"
    document = docx.Document()
    document.add_paragraph("Verbale di riunione.")
    document.core_properties.modified = datetime(2018, 11, 20, 8, 0)
    document.save(str(path))
    _with_mtime(path)

    result = reference_date(path, now=_NOW)
    assert result.source is DateSource.CONTENT_METADATA
    assert result.field == "docx:core_properties.modified"
    assert result.value == datetime(2018, 11, 20, 8, 0, tzinfo=timezone.utc)


def test_xlsx_workbook_properties(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    # openpyxl overwrites ``properties.modified`` with the current time on every
    # save, so setting it before saving would prove nothing: the workbook is
    # repacked with a known ``dcterms:modified`` instead, which is also closer to
    # the real case — a file written by Excel years ago.
    raw = tmp_path / "raw.xlsx"
    openpyxl.Workbook().save(str(raw))
    path = _repack_xlsx_modified(raw, tmp_path / "clienti.xlsx", "2017-02-01T15:45:00Z")
    _with_mtime(path)

    result = reference_date(path, now=_NOW)
    assert result.source is DateSource.CONTENT_METADATA
    assert result.field == "xlsx:properties.modified"
    assert result.value == datetime(2017, 2, 1, 15, 45, tzinfo=timezone.utc)


# --- the sanity guard --------------------------------------------------------


def test_metadata_in_the_future_is_rejected(tmp_path: Path) -> None:
    future = _NOW + timedelta(days=400)
    path = _pdf_with_metadata(
        tmp_path,
        {
            "modDate": future.strftime("D:%Y%m%d%H%M%SZ"),
            "creationDate": future.strftime("D:%Y%m%d%H%M%SZ"),
        },
    )
    result = reference_date(path, now=_NOW)
    assert result.source is DateSource.FILE_MTIME
    assert result.value == _MTIME


def test_epoch_metadata_is_rejected(tmp_path: Path) -> None:
    path = _pdf_with_metadata(
        tmp_path, {"modDate": "D:19700101000000Z", "creationDate": "D:19700101000000Z"}
    )
    result = reference_date(path, now=_NOW)
    assert result.source is DateSource.FILE_MTIME


def test_implausible_modification_falls_through_to_creation(tmp_path: Path) -> None:
    # The guard rejects a candidate, it does not abandon the metadata: a broken
    # modDate next to a sane creationDate must still yield the creation date.
    path = _pdf_with_metadata(
        tmp_path,
        {"modDate": "D:19700101000000Z", "creationDate": "D:20220915120000Z"},
    )
    result = reference_date(path, now=_NOW)
    assert result.source is DateSource.CONTENT_METADATA
    assert result.field == "pdf:creationDate"


# --- the PDF date parser, directly -------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("D:20190312101500+01'00'", datetime(2019, 3, 12, 9, 15, tzinfo=timezone.utc)),
        ("D:20190312101500-05'00'", datetime(2019, 3, 12, 15, 15, tzinfo=timezone.utc)),
        ("D:20190312101500Z", datetime(2019, 3, 12, 10, 15, tzinfo=timezone.utc)),
        ("D:20190312", datetime(2019, 3, 12, tzinfo=timezone.utc)),
        ("D:201903121015", datetime(2019, 3, 12, 10, 15, tzinfo=timezone.utc)),
        ("20190312101500", datetime(2019, 3, 12, 10, 15, tzinfo=timezone.utc)),
    ],
)
def test_parse_pdf_date_tolerates_truncation(raw: str, expected: datetime) -> None:
    assert _parse_pdf_date(raw) == expected


@pytest.mark.parametrize("raw", ["", "not a date", "D:2019", "D:20191332000000Z"])
def test_parse_pdf_date_rejects_garbage(raw: str) -> None:
    assert _parse_pdf_date(raw) is None
