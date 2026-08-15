"""The mess a real file share carries, planted on purpose and *declared*.

A corpus made only of well-formed documents proves the happy path and nothing
else. Real folders hold spreadsheets exported to CSV, screenshots, zip archives,
Office lock files left behind by a crash, a text file saved in the wrong
encoding, a PDF truncated by an interrupted copy. The scan must react to each in
a specific way — skip it silently, or record an isolated error and carry on —
and that reaction is what this module makes testable: every noise file is
planned with the
:class:`~pii_detection.evaluation.enterprise.types.Expectation` it must produce,
so a test asserts the exact behaviour instead of merely checking the run did not
crash.

Three groups, mirroring the three ways a file can be awkward:

* **unsupported formats** — extension outside
  :func:`~pii_detection.extraction.supported_suffixes`, so
  :func:`~pii_detection.registry.scan_folder.plan_folder` must leave them in
  ``skipped``, before any detector is even built;
* **pathological files** — supported extension, unreadable content, so
  :func:`~pii_detection.registry.scan_folder.ingest_folder` must put them in
  ``errors`` and keep going;
* **hostile names and paths** — spaces, accents, ``&``, very long names, deep
  nesting and a lowercase twin of an existing folder. These carry *valid*
  content: what they stress is the path-shaped identity, the folder-rule
  prefixes and the UI, not the readers.

The payload of a noise file is derived from its ``kind``, so the plan stays a
small comparable value and the bytes are produced only at write time.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime

from pii_detection.evaluation.enterprise.types import (
    DocumentSpec,
    Expectation,
    FolderSpec,
    SizeClass,
)

#: Prefix marking a :class:`~pii_detection.evaluation.enterprise.types.DocumentSpec`
#: as noise rather than a generated document.
NOISE_PREFIX = "noise:"

#: Unsupported extensions and the bytes to write for each. The content is
#: plausible (a real CSV, a real PNG header) so the file is only "wrong" in the
#: way that matters: the scanner does not read that format.
_UNSUPPORTED: dict[str, bytes] = {
    "csv": b"codice;descrizione;quantita\nART-101-A;guarnizione;250\n",
    "md": b"# Note di rilascio\n\n- corretto il calcolo dello sconto\n",
    "png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
    "zip": b"PK\x03\x04" + b"\x00" * 60,
    "pptx": b"PK\x03\x04" + b"\x00" * 96,
}

#: Pathological files: supported extension, content the reader cannot handle.
#: ``empty_txt`` is deliberately *not* one of them — an empty text file reads
#: fine and simply holds no PII, which is itself a case worth having.
_PATHOLOGICAL: dict[str, tuple[str, bytes]] = {
    "office_lock": ("docx", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 48),
    "empty_pdf": ("pdf", b""),
    "truncated_pdf": ("pdf", b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog"),
    "latin1_txt": ("txt", "Attività svolta in società: riepilogo.\n".encode("latin-1")),
}

#: Folders that exist only to be awkward. They hold ordinary documents: the
#: stress is on the names and the depth, not on the content.
HOSTILE_FOLDERS: tuple[FolderSpec, ...] = (
    FolderSpec(
        "Archivio/2019/Amministrazione/Contabilità & Bilanci/Chiuso/Scansioni",
        ("invoice", "meeting_minutes", "supplier_contract"),
        ("pdf", "txt"),
        weight=2,
        year=2019,
    ),
    FolderSpec(
        "Direzione/Verbali CdA/Riunione #12",
        ("meeting_minutes",),
        ("pdf", "docx"),
        weight=1,
        year=2022,
    ),
    # Lowercase twin of "HR"/"Risorse Umane": FolderRule.matches is
    # case-sensitive, so a rule on the uppercase folder must NOT catch this one.
    FolderSpec(
        "hr/contratti",
        ("hr_record", "payslip"),
        ("pdf", "docx"),
        weight=1,
        year=2023,
    ),
)

#: A file name long enough to bother a file system and a UI column, without
#: crossing the 255-byte limit most of them impose.
LONG_NAME = (
    "verbale-di-riunione-straordinaria-del-consiglio-di-amministrazione-"
    "con-allegati-tecnici-e-prospetti-economici-riclassificati-per-centro-"
    "di-costo-versione-definitiva-approvata"
)


#: Plausible base names for the unsupported attachments, by extension.
_UNSUPPORTED_NAMES: dict[str, str] = {
    "csv": "export_articoli",
    "md": "note_di_rilascio",
    "png": "schermata_errore",
    "zip": "allegati_pratica",
    "pptx": "presentazione_interna",
}


def noise_specs(
    folders: tuple[str, ...],
    rng: random.Random,
    stamp: Callable[[str], datetime],
) -> list[DocumentSpec]:
    """Plan one noise file of each kind, spread across the given folders.

    :param folders: folder paths (relative, POSIX) the noise is scattered over.
    :param rng: seeded RNG, so the placement is reproducible.
    :param stamp: maps a folder path to the modification time to apply — the
        same policy used for real documents, so noise files are not
        conspicuously fresher than the folder they sit in.
    :returns: the planned noise files, each carrying its expected outcome.
    :raises ValueError: if ``folders`` is empty.
    """
    if not folders:
        raise ValueError("noise needs at least one folder to live in")
    specs: list[DocumentSpec] = []
    for name in _UNSUPPORTED:
        folder = rng.choice(folders)
        specs.append(
            DocumentSpec(
                relative_path=f"{folder}/{_UNSUPPORTED_NAMES[name]}.{name}",
                kind=f"{NOISE_PREFIX}unsupported_{name}",
                size_class=SizeClass.SHORT,
                file_format=name,
                annotated_text="",
                modified_at=stamp(folder),
                expectation=Expectation.SKIPPED,
            )
        )
    for name, (suffix, payload) in _PATHOLOGICAL.items():
        folder = rng.choice(folders)
        filename = "~$contratto_2023.docx" if name == "office_lock" else f"{name}.{suffix}"
        specs.append(
            DocumentSpec(
                relative_path=f"{folder}/{filename}",
                kind=f"{NOISE_PREFIX}{name}",
                size_class=SizeClass.SHORT,
                file_format=suffix,
                annotated_text="",
                modified_at=stamp(folder),
                expectation=Expectation.ERROR,
            )
        )
    empty_folder = rng.choice(folders)
    specs.append(
        DocumentSpec(
            relative_path=f"{empty_folder}/appunti_vuoti.txt",
            kind=f"{NOISE_PREFIX}empty_txt",
            size_class=SizeClass.SHORT,
            file_format="txt",
            annotated_text="",
            modified_at=stamp(empty_folder),
            expectation=Expectation.SCANNABLE,  # readable, simply holds nothing
        )
    )
    return specs


def payload_for(kind: str) -> bytes:
    """Return the bytes of a noise file, derived from its ``kind``.

    :param kind: the ``noise:`` kind recorded in the
        :class:`~pii_detection.evaluation.enterprise.types.DocumentSpec`.
    :returns: the exact bytes to write.
    :raises KeyError: if the kind is not a known noise kind.
    """
    name = kind.removeprefix(NOISE_PREFIX)
    if name.startswith("unsupported_"):
        return _UNSUPPORTED[name.removeprefix("unsupported_")]
    if name == "empty_txt":
        return b""
    return _PATHOLOGICAL[name][1]


def is_noise(kind: str) -> bool:
    """:returns: whether a ``kind`` denotes a planted noise file."""
    return kind.startswith(NOISE_PREFIX)


__all__ = [
    "HOSTILE_FOLDERS",
    "LONG_NAME",
    "NOISE_PREFIX",
    "is_noise",
    "noise_specs",
    "payload_for",
]
