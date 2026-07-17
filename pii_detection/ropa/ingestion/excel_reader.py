# Responsabile per leggere i file excel e restituirli in un formato facilmente interpretabile per il parsing

# import openpyxml
from openpyxl import load_workbook
from dataclasses import dataclass

@dataclass(frozen=True)
class Rawtable:
    columns: tuple[str, ...]
    records: list[dict]

def read_records(path) -> list[dict]:
    # hardcoded path, will be smarter
    wb = load_workbook(path, read_only=True, data_only=True)
    # Set an active sheet
    ws = wb.active

    rows = ws.iter_rows(values_only=True)

    header = next(rows)

    records = [dict(zip(header, row)) for row in rows]
    
    wb.close()

    return Rawtable(columns=tuple(header), records=records)
