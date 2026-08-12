"""Generate example CNIL-format ROPA workbooks (ODS + XLSX) with Italian content,
then run the B1 ingestion + dictionary mapping on each and print the result.

The workbooks keep the CNIL structural labels in English (the normalizer keys off
them) but fill the categories in Italian, as a real Italian company using a CNIL
template would. Purpose: check the ingestion holds across formats and varied,
realistic wording.

Run from the project root:
    ./.venv/bin/python examples/ropa/demo.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from odf.opendocument import OpenDocumentSpreadsheet
from odf.table import Table, TableCell, TableRow
from odf.text import P
from openpyxl import Workbook

from pii_detection.ropa.ingestion.category_mapper import (
    CategoryMapper,
    build_dictionary_mapper,
    build_llm_category_mapper,
)
from pii_detection.ropa.ingestion.pipeline import ingest_file, map_categories
from pii_detection.ropa.repository import ROPARepository


def _mapper(kind: str) -> CategoryMapper:
    """Build the mapper selected on the command line (dictionary/llm/hybrid)."""
    if kind == "dictionary":
        return build_dictionary_mapper()
    if kind == "llm":
        return build_llm_category_mapper(use_fallback=False)
    return build_llm_category_mapper(use_fallback=True)

HERE = Path(__file__).resolve().parent

# (sheet, activity name, purpose, [(macro, description, retention), ...])
ACTIVITIES = [
    (
        "Gestione_HR",
        "Gestione del personale",
        "Amministrazione del rapporto di lavoro",
        [
            ("Dati identificativi", "nome, cognome e indirizzo di residenza", "5 anni"),
            ("Dati economici", "coordinate bancarie", "10 anni"),
            ("Contatti", "indirizzo email e numero di telefono", "a criterio"),
            ("Identificativi fiscali", "codice fiscale", "5 anni"),
        ],
    ),
    (
        "Marketing",
        "Newsletter e marketing",
        "Invio di comunicazioni commerciali",
        [
            ("Contatti", "email", "24 mesi"),
            ("Dati anagrafici", "dati anagrafici", "24 mesi"),
            ("Profilazione", "preferenze di acquisto profilate", "a criterio"),
        ],
    ),
    (
        "Videosorveglianza",
        "Videosorveglianza sede",
        "Sicurezza dei locali",
        [
            ("Immagini", "immagini delle telecamere", "7 giorni"),
            ("Rete", "indirizzo IP del dispositivo", "6 mesi"),
        ],
    ),
]


def _sheet_rows(name: str, purpose: str, categories: list[tuple[str, str, str]]) -> list[list[str]]:
    """Build the CNIL sheet grid for one processing activity."""
    rows = [
        ["Name of the processing operation", name],
        ["Main purpose", purpose],
        ["Categories of personal data", "Description", "Data retention period"],
    ]
    rows.extend([macro, description, retention] for macro, description, retention in categories)
    return rows


def write_xlsx(path: Path) -> None:
    """Write the example workbook as ``.xlsx`` (openpyxl)."""
    wb = Workbook()
    default = wb.active
    for sheet, name, purpose, categories in ACTIVITIES:
        ws = wb.create_sheet(sheet)
        for row in _sheet_rows(name, purpose, categories):
            ws.append(row)
    instructions = wb.create_sheet("Istruzioni")
    instructions.append(["Compilare una scheda per ogni trattamento."])
    if default is not None:
        wb.remove(default)
    wb.save(path)


def _ods_row(cells: list[str]) -> TableRow:
    """Build one ODS table row from a list of cell strings."""
    row = TableRow()
    for text in cells:
        cell = TableCell()
        cell.addElement(P(text=text))
        row.addElement(cell)
    return row


def write_ods(path: Path) -> None:
    """Write the example workbook as ``.ods`` (odfpy)."""
    doc = OpenDocumentSpreadsheet()
    for sheet, name, purpose, categories in ACTIVITIES:
        table = Table(name=sheet)
        for row in _sheet_rows(name, purpose, categories):
            table.addElement(_ods_row(row))
        doc.spreadsheet.addElement(table)
    instructions = Table(name="Istruzioni")
    instructions.addElement(_ods_row(["Compilare una scheda per ogni trattamento."]))
    doc.spreadsheet.addElement(instructions)
    doc.save(str(path))


def show(path: Path, kind: str) -> None:
    """Ingest a workbook, map its categories with the chosen mapper, print the tree."""
    db_url = f"sqlite:///{tempfile.mkdtemp()}/ropa.db"
    activities = ingest_file(path, db_url)
    map_categories(ROPARepository(db_url), _mapper(kind))
    print(f"\n=== {path.name} [{kind}]: {len(activities)} trattamenti (Istruzioni saltato) ===")
    for activity in ROPARepository(db_url).load():
        print(f"[{activity.id}] {activity.name}")
        for macro in activity.macro_categories:
            print(f"   macro {macro.raw_text!r}  retention_months={macro.retention_months}")
            for category in macro.categories:
                print(f"      - {category.raw_text!r:44} -> {category.pii_types}")


def main() -> None:
    """Generate both workbooks into ``corpus/ropa/`` and run the demo on each."""
    parser = argparse.ArgumentParser(description="Generate example ROPA workbooks and map them.")
    parser.add_argument(
        "--mapper",
        choices=["dictionary", "llm", "hybrid"],
        default="dictionary",
        help="which category mapper to run (default: dictionary)",
    )
    args = parser.parse_args()

    target = HERE.parents[1] / "corpus" / "ropa"  # repo-root/corpus/ropa (gitignored)
    target.mkdir(parents=True, exist_ok=True)
    ods, xlsx = target / "ropa_aziendale.ods", target / "ropa_aziendale.xlsx"
    write_ods(ods)
    write_xlsx(xlsx)
    print(f"generati:\n  {ods}\n  {xlsx}")
    show(ods, args.mapper)
    show(xlsx, args.mapper)


if __name__ == "__main__":
    main()
