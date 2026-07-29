"""CLI: extract and print the normalized text of a real document.

Lets you point the B3 reader at an actual PDF/Word/text file and see exactly the
text the detection layer would receive::

    python -m pii_detection.extraction path/to/document.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pii_detection.extraction.extractor import extract_document


def main(argv: list[str] | None = None) -> None:
    """Parse the path argument, extract the document and print its text.

    :param argv: argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        description="Extract and print a document's normalized text (block B3)."
    )
    parser.add_argument("path", type=Path, help="path to a .pdf, .docx or .txt file")
    args = parser.parse_args(argv)
    document = extract_document(args.path)
    print(f"# document_id: {document.document_id}\n")
    print(document.text)


if __name__ == "__main__":
    main()
