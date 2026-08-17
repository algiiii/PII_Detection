"""The two trees the corpus can take the shape of.

``realistic`` is the honest simulation: a company file share that owes nothing
to any register, because in a real organisation nobody has tidied the folders to
match the ROPA. It is the default, and it is the one that says whether the
system copes with what it will actually be pointed at.

``ropa`` is the instrumented tree: built around the activities of a **real**
register file, with the violations the compliance check is supposed to catch
planted on purpose —

* **orphan PII**: categories the register never declares (with
  ``corpus/ropa/ropa_aziendale.ods`` those are ``credit_card``, ``swiss_avs``
  and ``health_data``) dropped into folders that *are* covered by a rule, so the
  document is associated yet holds something nobody declared;
* **expired retention**: documents older than the retention their activity
  declares, via the modification time the registry reads back as the document's
  ``reference_date``;
* **uncovered folders**: PII-bearing documents under prefixes no folder rule
  reaches, which must stay unassociated.

The register is read with the ingestion pipeline the DPO uses
(:func:`~pii_detection.ropa.ingestion.pipeline.ingest_file` plus
:func:`~pii_detection.ropa.ingestion.pipeline.map_categories`), not with a
private parser: whatever the pipeline understands of a register is exactly what
the corpus is built against.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pii_detection.detection.config import default_config_dir, load_category_catalog
from pii_detection.evaluation.enterprise.noise import HOSTILE_FOLDERS
from pii_detection.evaluation.enterprise.types import FolderSpec

if TYPE_CHECKING:  # heavy ROPA imports only for typing: the realistic profile
    from pii_detection.ropa.types import ProcessingActivity  # needs none of them

#: The realistic profile: departments, years, projects. Weights decide how much
#: of the corpus each folder receives; ``year`` drives the modification times,
#: so the archive really is old.
REALISTIC_FOLDERS: tuple[FolderSpec, ...] = (
    FolderSpec("Amministrazione/Contabilità/2023",
               ("invoice", "payment_notice"), ("pdf", "docx"), weight=4, year=2023),
    FolderSpec("Amministrazione/Contabilità/2024",
               ("invoice", "payment_notice"), ("pdf", "docx"), weight=4, year=2024),
    FolderSpec("Amministrazione/Fatture fornitori",
               ("supplier_invoice", "supplier_contract"), ("pdf",), weight=3),
    FolderSpec("Risorse Umane/Contratti/2023",
               ("hr_contract", "hr_record"), ("pdf", "docx"), weight=4, year=2023),
    FolderSpec("Risorse Umane/Contratti/2024",
               ("hr_contract", "hr_record"), ("pdf", "docx"), weight=4, year=2024),
    FolderSpec("Risorse Umane/Buste paga/2023",
               ("payslip",), ("pdf",), weight=4, year=2023),
    FolderSpec("Risorse Umane/Selezione/Candidature 2024",
               ("cv", "interview_note"), ("pdf", "docx", "txt"), weight=3, year=2024),
    FolderSpec("Risorse Umane/Medicina del lavoro",
               ("occupational_health",), ("pdf",), weight=2),
    FolderSpec("Risorse Umane/Frontalieri",
               ("cross_border_notice",), ("pdf", "docx"), weight=2),
    FolderSpec("Risorse Umane/Procedure",
               ("policy",), ("pdf", "docx"), weight=2),
    FolderSpec("Commerciale/Clienti/Delta Servizi",
               ("client_letter", "quotation"), ("pdf", "docx"), weight=3),
    FolderSpec("Commerciale/Clienti/Nordest Logistica",
               ("client_letter", "payment_dispute"), ("pdf", "docx"), weight=3),
    FolderSpec("Commerciale/Anagrafiche",
               ("client_registry_export",), ("txt", "docx"), weight=2),
    FolderSpec("Commerciale/Listini",
               ("price_list",), ("pdf",), weight=3),
    FolderSpec("Marketing/Campagne/2024",
               ("campaign_brief", "newsletter_export"), ("pdf", "docx", "txt"),
               weight=3, year=2024),
    FolderSpec("IT/Log", ("access_log",), ("txt",), weight=3),
    FolderSpec("IT/Incidenti", ("incident_report",), ("txt", "docx"), weight=2),
    FolderSpec("IT/Documentazione", ("tech_manual",), ("pdf", "docx"), weight=3),
    FolderSpec("Legale/Contratti fornitori",
               ("supplier_contract",), ("pdf",), weight=3),
    FolderSpec("Legale/Privacy", ("policy",), ("pdf", "docx"), weight=2),
    FolderSpec("Direzione/Verbali/2024",
               ("meeting_minutes",), ("pdf", "docx"), weight=3, year=2024),
    FolderSpec("Archivio/2019/Risorse Umane",
               ("hr_contract", "payslip"), ("pdf", "docx"), weight=3, year=2019),
    FolderSpec("Archivio/2019/Commerciale",
               ("invoice", "client_letter"), ("pdf",), weight=2, year=2019),
) + HOSTILE_FOLDERS


@dataclass(frozen=True)
class DeclaredRegister:
    """What a real register declares, reduced to what the corpus needs.

    :ivar types: ``{activity_id: (pii_type, ...)}``.
    :ivar retention_months: ``{activity_id: months}`` — the **shortest** duration
        the activity declares, i.e. the first term that can be exceeded, or
        ``None`` when it states only criteria and nothing can be computed.
    """

    types: dict[str, tuple[str, ...]]
    retention_months: dict[str, int | None]


@dataclass(frozen=True)
class RetentionExpectation:
    """A folder that must make the retention check complain.

    :ivar prefix: folder prefix whose documents are old enough to be overdue.
    :ivar activity_id: the activity the folder rule associates them with.
    :ivar retention_months: the term declared by that activity.
    :ivar age_months: the approximate age the planted documents will have.
    """

    prefix: str
    activity_id: str
    retention_months: int
    age_months: int


@dataclass(frozen=True)
class RopaLayout:
    """The tree built around a register, plus what it must make the check say.

    :ivar folders: the folders to populate.
    :ivar rules: ``(prefix, activity_ids)`` pairs ready for
        :meth:`~pii_detection.registry.repository.PIIRepository.save_rule`.
    :ivar orphan_types: ``pii_type`` ids the register never declares, planted in
        covered folders so the compliance check must report them as orphan.
    :ivar retention: the folders whose documents must come back overdue. An
        activity that declares no computable duration contributes **no**
        expectation: the corpus states what it can prove, it does not guess.
    """

    folders: tuple[FolderSpec, ...]
    rules: tuple[tuple[str, tuple[str, ...]], ...]
    orphan_types: tuple[str, ...]
    retention: tuple[RetentionExpectation, ...] = ()


#: Nominal year of the archive folders, and the age that follows from it. Old
#: enough that any declared term shorter than this must be exceeded — which is
#: what makes the expectation below provable rather than hopeful.
_ARCHIVE_YEAR = 2019
_ARCHIVE_FOLDER = f"Archivio {_ARCHIVE_YEAR}"
_ARCHIVE_AGE_MONTHS = (date.today().year - _ARCHIVE_YEAR) * 12

#: Archetypes carrying the categories a register typically leaves undeclared,
#: by ``pii_type``. Planting an orphan means putting one of these documents in a
#: covered folder.
_ORPHAN_CARRIERS: dict[str, str] = {
    "credit_card": "payment_dispute",
    "swiss_avs": "cross_border_notice",
    "health_data": "occupational_health",
}

#: Archetypes to fill a covered folder with, by the ``pii_type`` its activity
#: declares — so a covered folder holds what its activity says it holds.
_DECLARED_CARRIERS: dict[str, tuple[str, ...]] = {
    "person_name": ("hr_record", "client_letter"),
    "address": ("hr_record", "invoice"),
    "iban": ("payslip", "invoice"),
    "email": ("client_letter", "newsletter_export"),
    "phone": ("client_letter", "quotation"),
    "italian_id": ("hr_record", "invoice"),
    "date_of_birth": ("newsletter_export", "cv"),
    "ip_address": ("access_log",),
}


def declared_types(activities: Sequence[ProcessingActivity]) -> dict[str, tuple[str, ...]]:
    """Map each activity id to the ``pii_type`` ids it declares.

    :param activities: the normalized processing activities of a register.
    :returns: ``{activity_id: (pii_type, ...)}``, in declaration order.
    """
    declared: dict[str, tuple[str, ...]] = {}
    for activity in activities:
        types: list[str] = []
        for macro in activity.macro_categories:
            for category in macro.categories:
                for pii_type in category.pii_types:
                    if pii_type not in types:
                        types.append(pii_type)
        declared[activity.id] = tuple(types)
    return declared


def declared_retentions(
    activities: Sequence[ProcessingActivity],
) -> dict[str, int | None]:
    """Map each activity id to the shortest retention it declares, in months.

    The shortest term is the one that trips first, so it is the one an aged
    document is guaranteed to exceed.

    :param activities: the normalized processing activities of a register.
    :returns: ``{activity_id: months}``, ``None`` when the activity declares no
        computable duration (only criteria).
    """
    shortest: dict[str, int | None] = {}
    for activity in activities:
        durations = [
            macro.retention_months
            for macro in activity.macro_categories
            if macro.retention_months is not None
        ]
        shortest[activity.id] = min(durations) if durations else None
    return shortest


def read_ropa(path: Path) -> DeclaredRegister:
    """Read a register file and return what each of its activities declares.

    Reuses the DPO's own ingestion path — sheets are normalized, then the
    dictionary mapper resolves the free-text categories onto ``pii_type`` ids —
    against a throwaway database, so nothing of the caller's state is touched.

    :param path: the ``.ods``/``.xlsx`` register.
    :returns: what the register declares: categories and retention per activity.
    :raises FileNotFoundError: if the register file does not exist.
    :raises ValueError: if the workbook holds no processing-activity sheet.
    """
    from pii_detection.ropa.ingestion.category_mapper import build_mapper
    from pii_detection.ropa.ingestion.pipeline import ingest_file, map_categories
    from pii_detection.ropa.repository import ROPARepository

    if not Path(path).exists():
        raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory() as tmp:
        db_url = f"sqlite:///{Path(tmp) / 'ropa_probe.db'}"
        activities = ingest_file(path, db_url)
        if not activities:
            raise ValueError(f"no processing activity found in {path}")
        repository = ROPARepository(db_url)
        map_categories(repository, build_mapper("dictionary"))
        activities = repository.load()
        return DeclaredRegister(
            types=declared_types(activities),
            retention_months=declared_retentions(activities),
        )


def ropa_layout(declared: DeclaredRegister) -> RopaLayout:
    """Build the instrumented tree around what a register declares.

    One covered folder per activity (its declared categories inside), one
    orphan-bearing folder per activity (a category the register never mentions,
    under the same covered prefix, so the document *is* associated), plus an
    explicitly uncovered branch no rule reaches.

    :param declared: what the register declares, from :func:`read_ropa`.
    :returns: folders, folder rules, the orphan types planted and the retention
        breaches the check must report.
    """
    catalog_file = default_config_dir() / "categories.yaml"
    catalog = {category.id for category in load_category_catalog(catalog_file)}
    all_declared = {
        pii_type for types in declared.types.values() for pii_type in types
    }
    orphan_types = tuple(
        pii_type for pii_type in _ORPHAN_CARRIERS if pii_type in catalog - all_declared
    )

    folders: list[FolderSpec] = []
    rules: list[tuple[str, tuple[str, ...]]] = []
    retention: list[RetentionExpectation] = []
    for activity_id, types in sorted(declared.types.items()):
        prefix = _folder_name(activity_id)
        kinds = tuple(
            kind
            for pii_type in types
            for kind in _DECLARED_CARRIERS.get(pii_type, ())
        ) or ("client_letter",)
        folders.append(
            FolderSpec(f"{prefix}/Documenti", tuple(dict.fromkeys(kinds)),
                       ("pdf", "docx", "txt"), weight=4)
        )
        # Old enough to trip the retention check of any activity that declares one.
        folders.append(
            FolderSpec(f"{prefix}/{_ARCHIVE_FOLDER}", tuple(dict.fromkeys(kinds)),
                       ("pdf", "docx"), weight=2, year=_ARCHIVE_YEAR)
        )
        # Covered, associated, and holding no PII at all: the case where every
        # declared category comes back "never found" in the verdict.
        folders.append(
            FolderSpec(f"{prefix}/Procedure", ("policy", "meeting_minutes"),
                       ("pdf", "docx"), weight=3)
        )
        if orphan_types:
            folders.append(
                FolderSpec(
                    f"{prefix}/Materiale non dichiarato",
                    tuple(_ORPHAN_CARRIERS[pii_type] for pii_type in orphan_types),
                    ("pdf", "docx"),
                    weight=2,
                )
            )
        rules.append((prefix, (activity_id,)))
        term = declared.retention_months.get(activity_id)
        if term is not None and _ARCHIVE_AGE_MONTHS > term:
            retention.append(
                RetentionExpectation(
                    prefix=f"{prefix}/{_ARCHIVE_FOLDER}",
                    activity_id=activity_id,
                    retention_months=term,
                    age_months=_ARCHIVE_AGE_MONTHS,
                )
            )
    folders.append(
        FolderSpec("Varie/Senza regola", ("client_letter", "hr_record"),
                   ("pdf", "docx", "txt"), weight=3)
    )
    folders.append(
        FolderSpec("Varie/Documentazione", ("tech_manual", "policy"), ("pdf",), weight=2)
    )
    return RopaLayout(
        tuple(folders) + HOSTILE_FOLDERS, tuple(rules), orphan_types, tuple(retention)
    )


def _folder_name(activity_id: str) -> str:
    """Turn an activity id into a plausible folder name (``a-b`` → ``A b``)."""
    words = activity_id.replace("-", " ").strip()
    return words[:1].upper() + words[1:]


__all__ = [
    "REALISTIC_FOLDERS",
    "DeclaredRegister",
    "RetentionExpectation",
    "RopaLayout",
    "declared_retentions",
    "declared_types",
    "read_ropa",
    "ropa_layout",
]
