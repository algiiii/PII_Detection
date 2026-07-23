"""Populate a ROPA database from a CNIL spreadsheet (ODS or Excel).

Usage::

    python -m pii_detection.ropa.ingestion <register.ods> --db sqlite:///ropa.db

The database URL defaults to the ``ROPA_DB_URL`` environment variable (then to
``sqlite:///ropa.db``), so the same command runs locally and in a container.
"""

from __future__ import annotations

import argparse
import os

from sqlalchemy.exc import IntegrityError

from pii_detection.ropa.ingestion.category_mapper import build_dictionary_mapper
from pii_detection.ropa.ingestion.pipeline import ingest_file, map_categories
from pii_detection.ropa.repository import ROPARepository


def main() -> None:
    """Parse CLI options, ingest the spreadsheet and report how many activities
    were saved."""
    parser = argparse.ArgumentParser(
        description="Ingest a CNIL ROPA spreadsheet (ODS or Excel) into the database."
    )
    parser.add_argument("path", help="path to the .ods or .xlsx ROPA register")
    parser.add_argument(
        "--db",
        default=os.environ.get("ROPA_DB_URL", "sqlite:///ropa.db"),
        help="database URL (default: $ROPA_DB_URL or sqlite:///ropa.db)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="wipe the existing register before ingesting (destructive)",
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help="after ingesting, resolve declared categories onto pii_type via the "
        "deterministic dictionary mapper (leaves unresolved ones for the DPO)",
    )
    args = parser.parse_args()

    try:
        activities = ingest_file(args.path, args.db, replace=args.replace)
    except IntegrityError:
        parser.error(
            f"{args.db} already contains activities with these ids; "
            "re-run with --replace to overwrite, or point --db to a fresh file."
        )

    print(f"ingested {len(activities)} activities into {args.db}")

    if args.map:
        split = map_categories(ROPARepository(args.db), build_dictionary_mapper())
        print(f"mapped {split} declared categories")


if __name__ == "__main__":
    main()
