from pathlib import Path
from openpyxl import load_workbook
from odf.opendocument import load as load_ods
from odf.table import Table, TableRow, TableCell
from odf.text import P


def read_sheet(path: str | Path, sheet_name: str) -> list[list[str]]:
    """Read one sheet of a spreadsheet into a grid of cell strings.

    Dispatches on the file extension: ``.ods`` via odfpy, ``.xlsx``/``.xlsm``
    via openpyxl. Both back-ends yield the same shape, so the normalizer is
    format-agnostic.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".ods":
        return _read_ods(path, sheet_name)
    if suffix in (".xlsx", ".xlsm"):
        return _read_xlsx(path, sheet_name)
    raise ValueError(f"unsupported spreadsheet format: {suffix!r}")


def _read_xlsx(path, sheet_name):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [("" if v is None else str(v)).strip() for v in row]
            while cells and cells[-1] == "":
                cells.pop()
            rows.append(cells)
        return rows
    finally:
        wb.close()


def _read_ods(path, sheet_name):
    doc = load_ods(path)
    for table in doc.spreadsheet.getElementsByType(Table):
        if table.getAttribute("name") != sheet_name:
            continue
        rows = []
        for tr in table.getElementsByType(TableRow):
            cells = []
            for tc in tr.getElementsByType(TableCell):
                rep = int(tc.getAttribute("numbercolumnsrepeated") or 1)
                text = "".join(str(p) for p in tc.getElementsByType(P)).strip()
                cells.extend([text] * min(rep, 40))
            while cells and cells[-1] == "":
                cells.pop()
            rows.append(cells)
        return rows
    raise KeyError(f"sheet not found: {sheet_name!r}")