"""CLI for the compliance block: B6 association and B7 check.

Two subcommands::

    python -m pii_detection.compliance assign <document_id> --activities id1,id2
    python -m pii_detection.compliance check  <document_id> [--include-proposed]

``assign`` records which processing activities a document belongs to (explicit,
DPO-driven, B6). ``check`` compares the document's detected PII against those
activities' declared categories and prints the verdict (B7). Database URLs are
read from the environment (``ROPA_DB_URL``, ``PII_DB_URL``), so the same command
runs locally and in a container.
"""

from __future__ import annotations

import argparse

from pii_detection.compliance.assign import ExplicitAssigner, persist_assignment
from pii_detection.compliance.checker import check_document
from pii_detection.compliance.types import format_report
from pii_detection.registry.repository import PIIRepository
from pii_detection.ropa.repository import ROPARepository


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the selected compliance subcommand.

    :param argv: argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        description="Associate a document with processing activities (B6) and check "
        "its compliance (B7)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    assign_parser = subparsers.add_parser(
        "assign", help="assign a document to the activities it belongs to (B6)"
    )
    assign_parser.add_argument("document_id", help="id of an already-ingested document")
    assign_parser.add_argument(
        "--activities",
        required=True,
        help="comma-separated processing-activity ids from the ROPA",
    )

    check_parser = subparsers.add_parser(
        "check", help="check a document's PII against its declared activities (B7)"
    )
    check_parser.add_argument("document_id", help="id of an already-ingested document")
    check_parser.add_argument(
        "--include-proposed",
        action="store_true",
        help="count PROPOSED category mappings too, not only DPO-confirmed ones",
    )

    args = parser.parse_args(argv)
    registry = PIIRepository()

    if args.command == "assign":
        ids = [part.strip() for part in args.activities.split(",") if part.strip()]
        assigner = ExplicitAssigner(ids)
        persisted = persist_assignment(registry, args.document_id, assigner)
        print(f"Document '{args.document_id}' assigned to: {', '.join(persisted)}")
    else:  # check
        ropa = ROPARepository()
        report = check_document(
            args.document_id,
            ropa=ropa,
            registry=registry,
            include_proposed=args.include_proposed,
        )
        print(format_report(report))


if __name__ == "__main__":
    main()


__all__ = ["main"]
