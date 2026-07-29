"""Render the synthetic corpus to real document files (Tier 2 input).

Takes the generated annotated documents, strips the markers to the clean text a
real file would contain, and writes PDF/DOCX files plus a value-based gold file
(``gold.jsonl``). The annotated text stays the single source of truth: the
rendered files and the gold are both *derived* here, in one pass, so they cannot
drift. A PDF cannot carry inline markers, hence the gold lives beside the files
as ``(pii_type, value)`` pairs — the granularity the end-to-end scorer needs once
extraction has shifted the character offsets.

Rendering is intentionally simple (single-column, born-digital): it exercises the
extract->detect wiring and born-digital extraction, **not** real-world layout
chaos (multi-column, tables, scans), which would need OCR/layout tools.

Needs ``[eval]`` (``fpdf2``) for PDF and ``[extraction]`` (``python-docx``) for
DOCX; both are imported lazily, so rendering one format works without the other.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from pii_detection.evaluation.corpus import parse_annotated_text
from pii_detection.evaluation.corpus_generator import default_generated_dir, generate_documents


#: Typographic characters fpdf2's latin-1 core font cannot encode, mapped to
#: ASCII. Only prose punctuation is affected; PII values are already latin-1-safe.
_PUNCT_MAP = str.maketrans(
    {"—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "..."}
)


def _latin1_safe(line: str) -> str:
    """Reduce a line to what fpdf2's core font can render (prose only)."""
    return line.translate(_PUNCT_MAP).encode("latin-1", "replace").decode("latin-1")


def render_pdf(text: str, path: Path) -> None:
    """Render plain text into a single-column PDF (fpdf2, latin-1 core font).

    :param text: clean text to lay out, one source line per line.
    :param path: destination ``.pdf`` path.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_margins(20, 20, 20)  # explicit margins: default ones make wrapped
    pdf.add_page()               # lines extract back one word per line (fpdf2 quirk)
    pdf.set_font("Helvetica", size=12)
    # One multi_cell for the whole text: fpdf2 handles newlines and wraps long
    # lines naturally, so the PDF extracts back as continuous text.
    pdf.multi_cell(0, 8, _latin1_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.output(str(path))


def render_docx(text: str, path: Path) -> None:
    """Render plain text into a ``.docx``, one paragraph per line (python-docx).

    :param text: clean text to lay out.
    :param path: destination ``.docx`` path.
    """
    from docx import Document

    document = Document()
    for line in text.split("\n"):
        document.add_paragraph(line)
    document.save(str(path))


def _gold_for(document_id: str, annotated: str) -> dict[str, object]:
    """Build the value-based gold record of one document from its annotation."""
    parsed = parse_annotated_text(document_id, annotated)
    return {
        "document_id": document_id,
        "pii": [
            {"pii_type": span.pii_type, "value": parsed.text[span.start : span.end]}
            for span in parsed.spans
        ],
    }


def default_rendered_dir() -> Path:
    """:returns: the default output dir ``documents_generated/rendered``."""
    return default_generated_dir() / "rendered"


def render_corpus(out_dir: Path, n: int, seed: int, *, formats: Iterable[str]) -> Path:
    """Render ``n`` documents to the requested formats and write the gold file.

    :param out_dir: destination directory (created if missing).
    :param n: number of documents to render.
    :param seed: reproducibility seed (matches the generator).
    :param formats: any of ``"pdf"``, ``"docx"``.
    :returns: the path of the written ``gold.jsonl``.
    """
    formats = set(formats)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_path = out_dir / "gold.jsonl"
    with gold_path.open("w", encoding="utf-8") as gold_file:
        for doc_id, annotated in generate_documents(n, seed):
            clean = parse_annotated_text(doc_id, annotated).text
            if "pdf" in formats:
                render_pdf(clean, out_dir / f"{doc_id}.pdf")
            if "docx" in formats:
                render_docx(clean, out_dir / f"{doc_id}.docx")
            gold_file.write(json.dumps(_gold_for(doc_id, annotated), ensure_ascii=False) + "\n")
    return gold_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: render the corpus to disk.

    :param argv: argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Render the synthetic corpus to PDF/DOCX.")
    parser.add_argument("--out", type=Path, default=default_rendered_dir())
    parser.add_argument("--n", type=int, default=60, help="number of documents")
    parser.add_argument("--seed", type=int, default=42, help="reproducibility seed")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["pdf", "docx"],
        default=["pdf", "docx"],
        help="output formats to render",
    )
    args = parser.parse_args(argv)
    gold = render_corpus(args.out, args.n, args.seed, formats=args.formats)
    print(f"rendered {args.n} documents ({', '.join(args.formats)}) to {args.out}")
    print(f"gold written to {gold}")


if __name__ == "__main__":
    main()


__all__ = [
    "render_pdf",
    "render_docx",
    "render_corpus",
    "default_rendered_dir",
    "main",
]
