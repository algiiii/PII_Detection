"""Tests for the enterprise corpus generator.

Most of them run against the *plan* — a pure value — so the properties that make
the corpus worth generating (reproducibility, the length mix, the share of
documents without PII, category coverage, folder rules that really match) are
asserted in milliseconds without writing a file.

The few that do touch the disk check the contract with the scanner: the
``document_id`` computed by :func:`~pii_detection.registry.scan_folder.plan_folder`
is the one in the gold, the planted noise produces exactly the outcome it
declared, modification times survive, and a long PDF still gives every gold
value back after extraction.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("faker")
pytest.importorskip("fpdf")

from pii_detection.detection.config import (  # noqa: E402
    default_config_dir,
    load_category_catalog,
)
from pii_detection.detection.types import (  # noqa: E402
    ConfirmationLevel,
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    PIIMatch,
    TextSpan,
)
from pii_detection.extraction.dates import DateSource, ReferenceDate  # noqa: E402
from pii_detection.evaluation.corpus import parse_annotated_text  # noqa: E402
from pii_detection.evaluation.corpus_generator import PIIValueFactory  # noqa: E402
from pii_detection.evaluation.enterprise.builder import (  # noqa: E402
    PII_FREE_SHARE,
    load_manifest,
    load_sources,
    plan_corpus,
    write_corpus,
)
from pii_detection.evaluation.enterprise.content import (  # noqa: E402
    ARCHETYPES,
    SIZE_LINES,
    build_body,
)
from pii_detection.evaluation.enterprise.noise import is_noise  # noqa: E402
from pii_detection.evaluation.enterprise.profiles import (  # noqa: E402
    DeclaredRegister,
    ropa_layout,
)
from pii_detection.evaluation.enterprise.types import (  # noqa: E402
    CorpusPlan,
    DocumentSpec,
    Expectation,
    Profile,
    SizeClass,
)
from pii_detection.evaluation.render import load_gold  # noqa: E402
from pii_detection.extraction import supported_suffixes  # noqa: E402
from pii_detection.registry.folder_rules import match_activities  # noqa: E402
from pii_detection.registry.types import FolderRule  # noqa: E402


class _NullDetector:
    """A detector that never fires: the scan is exercised, not the detection."""

    detector_id = "fake.null"
    detector_kind = DetectorKind.REGEX

    def detect(self, text: str) -> list[PIICandidate]:
        return []


# --- the plan, as a pure value ------------------------------------------------


def test_plan_is_reproducible_and_seed_dependent() -> None:
    assert plan_corpus(30, seed=7) == plan_corpus(30, seed=7)
    assert plan_corpus(30, seed=7) != plan_corpus(30, seed=8)


def test_size_mix_covers_every_class_and_respects_its_band() -> None:
    plan = plan_corpus(400, seed=3)
    documents = plan.scannable()
    by_class = Counter(d.size_class for d in documents)
    assert set(by_class) == set(SizeClass), "the rare classes must not vanish"
    assert by_class[SizeClass.SHORT] > by_class[SizeClass.HUGE]
    for document in documents:
        if not document.annotated_text:
            continue
        lines = len(document.annotated_text.splitlines())
        low, high = SIZE_LINES[document.size_class]
        # The blocks that must appear can push a short body slightly past its
        # band; what matters is that the classes stay far apart in magnitude.
        assert lines >= low
        assert lines <= high * 1.5


def test_about_a_third_of_the_documents_carry_no_pii() -> None:
    plan = plan_corpus(300, seed=11)
    documents = [d for d in plan.scannable() if d.annotated_text]
    without = [
        d for d in documents if not parse_annotated_text(d.relative_path, d.annotated_text).spans
    ]
    share = len(without) / len(documents)
    assert PII_FREE_SHARE - 0.12 <= share <= PII_FREE_SHARE + 0.12
    assert all(not ARCHETYPES[d.kind].has_pii for d in without)


def test_every_catalog_category_appears_in_the_corpus() -> None:
    catalog = {category.id for category in load_category_catalog(
        default_config_dir() / "categories.yaml"
    )}
    plan = plan_corpus(300, seed=5)
    found = {
        span.pii_type
        for document in plan.documents
        for span in parse_annotated_text(
            document.relative_path, document.annotated_text
        ).spans
    }
    assert catalog == found


def test_pii_free_archetypes_carry_distractors_but_no_markers() -> None:
    factory = PIIValueFactory(2)
    body = build_body(ARCHETYPES["meeting_minutes"], factory, 60)
    assert "{{" not in body
    assert any(char.isdigit() for char in body), "distractors are the point"


def test_noise_is_planned_with_the_outcome_it_must_produce() -> None:
    plan = plan_corpus(40, seed=1)
    by_expectation = Counter(d.expectation for d in plan.documents)
    assert by_expectation[Expectation.SKIPPED] >= 5
    assert by_expectation[Expectation.ERROR] >= 4
    # The empty text file is noise too, yet it is *scannable*: it reads fine and
    # simply holds no PII, so it belongs in the gold with an empty record.
    empty = [d for d in plan.documents if d.kind.endswith("empty_txt")]
    assert [d.expectation for d in empty] == [Expectation.SCANNABLE]

    quiet = plan_corpus(40, seed=1, with_noise=False)
    assert len(quiet.documents) == 40
    assert not any(is_noise(d.kind) for d in quiet.documents)


def _declared() -> DeclaredRegister:
    return DeclaredRegister(
        types={
            "gestione-del-personale": ("person_name", "address", "iban"),
            "newsletter-e-marketing": ("email", "date_of_birth"),
        },
        retention_months={"gestione-del-personale": 60, "newsletter-e-marketing": None},
    )


def test_ropa_layout_plants_only_undeclared_types_and_matching_rules() -> None:
    declared = _declared()
    layout = ropa_layout(declared)
    assert set(layout.orphan_types) == {"credit_card", "swiss_avs", "health_data"}
    assert not set(layout.orphan_types) & {
        t for types in declared.types.values() for t in types
    }

    rules = [FolderRule(prefix=prefix, activity_ids=list(ids)) for prefix, ids in layout.rules]
    covered = [f.path for f in layout.folders if f.path.startswith("Gestione del personale")]
    assert covered, "each activity gets its own branch"
    for path in covered:
        assert match_activities(f"{path}/documento.pdf", rules) == ["gestione-del-personale"]
    assert match_activities("Varie/Senza regola/documento.pdf", rules) == []


def test_retention_expectations_only_where_a_term_is_computable() -> None:
    # An activity that declares a criterion instead of a duration yields no
    # expectation: the corpus asserts what it can prove, it does not guess.
    layout = ropa_layout(_declared())

    assert [e.activity_id for e in layout.retention] == ["gestione-del-personale"]
    (expectation,) = layout.retention
    assert expectation.retention_months == 60
    assert expectation.age_months > 60  # the archive folder really is older
    assert expectation.prefix.startswith("Gestione del personale/Archivio")
    # ...and the folder it names is one the layout actually populates.
    assert any(folder.path == expectation.prefix for folder in layout.folders)


def test_hostile_lowercase_twin_is_not_caught_by_the_uppercase_rule() -> None:
    # FolderRule matching is case-sensitive: the corpus plants the twin so the
    # limitation is visible, rather than discovered in production.
    rules = [FolderRule(prefix="HR", activity_ids=["a"])]
    assert match_activities("hr/contratti/x.pdf", rules) == []


# --- the corpus on disk -------------------------------------------------------


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, CorpusPlan]:
    """Write a small corpus once and share it across the disk-bound tests."""
    root = tmp_path_factory.mktemp("corpus")
    plan = plan_corpus(24, seed=4)
    write_corpus(plan, root)
    return root, plan


def test_gold_ids_are_the_ids_the_scanner_computes(
    written: tuple[Path, CorpusPlan]
) -> None:
    from pii_detection.registry.scan_folder import plan_folder

    root, plan = written
    folder_plan = plan_folder(root / "tree")
    scannable = {document_id for _, document_id in folder_plan.scannable}
    skipped = {p.relative_to(root / "tree").as_posix() for p in folder_plan.skipped}
    gold = load_gold(root / "gold.jsonl")

    declared_skipped = {
        d.relative_path for d in plan.documents if d.expectation is Expectation.SKIPPED
    }
    declared_errors = {
        d.relative_path for d in plan.documents if d.expectation is Expectation.ERROR
    }
    assert skipped == declared_skipped
    # A pathological file has a supported extension: the planner still lists it,
    # it is the reader that will fail on it.
    assert scannable == set(gold) | declared_errors


def test_pathological_files_fail_in_isolation(written: tuple[Path, CorpusPlan]) -> None:
    from pii_detection.registry.repository import PIIRepository
    from pii_detection.registry.scan_folder import ingest_folder

    root, plan = written
    repository = PIIRepository(f"sqlite:///{root}/registry.db")
    result = ingest_folder(
        root / "tree", _NullDetector(), _NullDetector(), repository=repository, prune=False
    )
    failed = {path.relative_to(root / "tree").as_posix() for path, _ in result.errors}
    declared = {
        d.relative_path for d in plan.documents if d.expectation is Expectation.ERROR
    }
    assert failed == declared
    assert result.scanned == len(load_gold(root / "gold.jsonl"))


def test_modification_times_are_applied(written: tuple[Path, CorpusPlan]) -> None:
    import os

    root, _ = written
    for record in load_manifest(root / "manifest.jsonl"):
        path = root / "tree" / str(record["document_id"])
        on_disk = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        assert on_disk == datetime.fromisoformat(str(record["modified_at"]))


def test_manifest_and_sources_cover_every_document(
    written: tuple[Path, CorpusPlan]
) -> None:
    root, plan = written
    manifest = load_manifest(root / "manifest.jsonl")
    assert len(manifest) == len(plan.documents)
    sources = load_sources(root / "sources.jsonl")
    for document in plan.scannable():
        if document.annotated_text:
            assert sources[document.relative_path] == document.annotated_text


def test_nothing_outside_the_tree_is_scannable(written: tuple[Path, CorpusPlan]) -> None:
    """Pointing the scan one level too high must not ingest the support files.

    They used to be a folder of ``<name>.pdf.txt`` sources beside the tree: a
    scan aimed at the root swallowed them as if they were the corpus, and every
    document in the registry came back as a ``.txt``.
    """
    root, _ = written
    supported = supported_suffixes()
    outside = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (root / "tree") not in path.parents
        and path.suffix.lower() in supported
    ]
    assert outside == []


def test_rewriting_a_smaller_corpus_leaves_nothing_behind(tmp_path: Path) -> None:
    """A regenerated corpus must describe itself: no orphans from the last run."""
    write_corpus(plan_corpus(30, seed=1), tmp_path)
    write_corpus(plan_corpus(10, seed=2), tmp_path)

    on_disk = {
        path.relative_to(tmp_path / "tree").as_posix()
        for path in (tmp_path / "tree").rglob("*")
        if path.is_file()
    }
    declared = {record["document_id"] for record in load_manifest(tmp_path / "manifest.jsonl")}
    assert on_disk == declared


def test_long_pdf_gives_back_every_gold_value(tmp_path: Path) -> None:
    """A 3-8 page PDF must survive extraction, not just a one-page note."""
    import re

    pytest.importorskip("fitz")
    from pii_detection.extraction import extract_document

    factory = PIIValueFactory(9)
    body = build_body(ARCHETYPES["hr_contract"], factory, 200)
    spec = DocumentSpec(
        relative_path="HR/contratto-lungo.pdf",
        kind="hr_contract",
        size_class=SizeClass.LONG,
        file_format="pdf",
        annotated_text=body,
        modified_at=datetime(2024, 5, 1, tzinfo=timezone.utc),
        expectation=Expectation.SCANNABLE,
    )
    write_corpus(CorpusPlan(Profile.REALISTIC, 9, (spec,)), tmp_path)

    extracted = re.sub(r"\s+", " ", extract_document(tmp_path / "tree" / spec.relative_path).text)
    parsed = parse_annotated_text(spec.relative_path, body)
    assert len(parsed.spans) >= 4
    for span in parsed.spans:
        value = re.sub(r"\s+", " ", parsed.text[span.start : span.end])
        assert value in extracted


def test_check_retention_catches_silence_and_over_reporting(tmp_path: Path) -> None:
    # The verifier's own two failure modes, on a hand-built registry: a planted
    # breach that never comes back, and a breach reported where nothing was
    # planted. Both must be named.
    from pii_detection.evaluation.enterprise.verify import check_retention
    from pii_detection.registry.repository import PIIRepository
    from pii_detection.ropa.repository import ROPARepository
    from pii_detection.ropa.types import (
        DeclaredCategory,
        DeclaredMacroCategory,
        MappingState,
        ProcessingActivity,
    )

    ropa = ROPARepository(url=f"sqlite:///{tmp_path}/ropa.db")
    registry = PIIRepository(url=f"sqlite:///{tmp_path}/pii.db")
    ropa.save(
        [
            ProcessingActivity(
                id="paghe",
                name="Paghe",
                purpose="p",
                macro_categories=[
                    DeclaredMacroCategory(
                        raw_text="Anagrafica",
                        retention_text="1 anno",
                        retention_months=12,
                        categories=[
                            DeclaredCategory(
                                raw_text="iban",
                                pii_types=["iban"],
                                mapping_state=MappingState.CONFIRMED,
                            )
                        ],
                    )
                ],
            )
        ]
    )

    old = datetime.now(timezone.utc) - timedelta(days=3000)
    for document_id in ("Paghe/Archivio 2019/a.pdf", "Paghe/Documenti/nuovo.pdf"):
        registry.record_scan(
            document_id,
            [_iban_match()],
            reference_date=ReferenceDate(
                value=old, source=DateSource.FILE_MTIME, field="fs:mtime"
            ),
        )
        registry.assign_activities(document_id, ["paghe"])

    result = check_retention(
        [{"prefix": "Paghe/Archivio 2019"}], registry=registry, ropa=ropa
    )

    assert result.silent == ()  # the planted folder did come back
    assert result.unexpected == ("Paghe/Documenti/nuovo.pdf",)  # ...and so did a recent one
    assert result.clean is False


def _iban_match() -> PIIMatch:
    provenance = DetectionProvenance("det.x", DetectorKind.REGEX, "iban", 0.9)
    return PIIMatch(
        span=TextSpan(0, 10),
        text="?",
        pii_type="iban",
        confidence=0.9,
        confirmation_level=ConfirmationLevel.SINGLE_SOURCE,
        sources=[provenance],
        document_id="ignored",
    )
