"""Quick manual check that the ROPA reader parses ROPA.xlsx correctly.

Not a pytest test (no ``test_`` prefix, so pytest ignores it): a throwaway
script to eyeball what ``excel_reader.read_records`` returns. It drives the REAL
reader code, not a separate openpyxl read. Run it from the repo root with:

    ./.venv/bin/python -m pii_detection.tests.check_ropa

Note: ``activity_id`` is intentionally NOT a column in the sheet — it is the one
field the normalizer will generate. Everything else comes straight from Excel.
"""

from pathlib import Path

from pii_detection.ropa.ingestion.excel_reader import read_records

XLSX = Path(__file__).with_name("ROPA.xlsx")


def main() -> None:
    table = read_records(XLSX)
    print(f"columns ({len(table.columns)}): {table.columns}")
    print(f"{len(table.records)} record(s) read from {XLSX.name}\n")
    for i, record in enumerate(table.records, start=1):
        print(f"--- record #{i} ---")
        for key, value in record.items():
            print(f"  {key}: {value!r}")
        print()


if __name__ == "__main__":
    main()
