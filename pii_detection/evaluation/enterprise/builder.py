"""From a seed to a corpus: plan it as a value, then write it once.

:func:`plan_corpus` decides everything — which file exists where, its content,
its length, its modification time, its expected fate — and returns it as an
immutable :class:`~pii_detection.evaluation.enterprise.types.CorpusPlan`. No
directory is touched, so the assertions that matter (reproducibility, the size
mix, the share of documents without PII, category coverage, the folder rules
actually matching) run in milliseconds against a value.

:func:`write_corpus` then materialises the plan in a single pass, emitting four
artefacts that cannot drift because they come from the same source text:

``tree/``
    the only thing to point a scan at;
``gold.jsonl``
    ``{document_id, pii: [...]}``, with ``document_id`` **verbatim** the path
    relative to ``tree/`` that
    :func:`~pii_detection.registry.scan_folder.ingest_folder` computes;
``manifest.jsonl``
    the same ids plus archetype, size class, format, modification time and
    expected outcome — what a test asserts the scan against;
``sources.jsonl``
    the annotated text of each document, kept for inspection and for
    value-level scoring.

The support files live outside ``tree/`` **and carry an extension no reader
supports**. Both halves matter: inside the tree they would be scanned as
documents, and as a folder of ``.txt`` files beside it they would be scanned the
moment someone pointed the scan one level too high — 200 annotated sources
ingested as if they were the corpus, every one of them named ``*.pdf.txt``.
Keeping them in a single ``.jsonl`` makes that mistake impossible instead of
merely documented.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pii_detection.evaluation.corpus import parse_annotated_text
from pii_detection.evaluation.corpus_generator import PIIValueFactory
from pii_detection.evaluation.enterprise import noise as noise_module
from pii_detection.evaluation.enterprise.content import ARCHETYPES, build_body, target_lines
from pii_detection.evaluation.enterprise.profiles import (
    REALISTIC_FOLDERS,
    RetentionExpectation,
    read_ropa,
    ropa_layout,
)
from pii_detection.evaluation.enterprise.types import (
    CorpusPlan,
    DocumentSpec,
    Expectation,
    FolderSpec,
    Profile,
    SizeClass,
)
from pii_detection.evaluation.render import gold_record, render_docx, render_pdf

#: Share of the corpus per size class. Most real documents are short; the few
#: very long ones are where extraction and detection actually get stressed.
SIZE_MIX: tuple[tuple[SizeClass, float], ...] = (
    (SizeClass.SHORT, 0.50),
    (SizeClass.MEDIUM, 0.35),
    (SizeClass.LONG, 0.13),
    (SizeClass.HUGE, 0.02),
)

#: Target share of documents holding no PII at all. A company share is mostly
#: harmless paperwork, and only these documents can reveal a false positive.
PII_FREE_SHARE = 0.35

#: Fallback reference date, so a plan built with no clock is still deterministic.
_DEFAULT_TODAY = datetime(2026, 1, 15, 9, 30, tzinfo=timezone.utc)


@dataclass(frozen=True)
class WriteSummary:
    """What :func:`write_corpus` put on disk.

    :ivar root: the directory the artefacts were written to.
    :ivar tree: the directory to point a scan at.
    :ivar written: number of files written, noise included.
    :ivar by_format: file count per extension.
    :ivar by_size: file count per size class (documents only).
    :ivar pii_free: number of documents with an empty gold record.
    :ivar total_bytes: size of ``tree/`` on disk.
    """

    root: Path
    tree: Path
    written: int
    by_format: dict[str, int]
    by_size: dict[str, int]
    pii_free: int
    total_bytes: int


def _size_sequence(n: int, rng: random.Random) -> list[SizeClass]:
    """Draw ``n`` size classes honouring :data:`SIZE_MIX`, then shuffle them.

    Quotas rather than independent draws: on 300 documents an independent 2%
    coin flip could easily yield zero huge files, and the corpus would quietly
    stop testing the case it exists for.

    :param n: number of documents.
    :param rng: seeded RNG.
    :returns: one size class per document, in assignment order.
    """
    sizes: list[SizeClass] = []
    for size_class, share in SIZE_MIX:
        sizes.extend([size_class] * int(n * share))
    while len(sizes) < n:
        sizes.append(SizeClass.SHORT)
    rng.shuffle(sizes)
    return sizes[:n]


def _folder_sequence(folders: tuple[FolderSpec, ...], n: int) -> list[FolderSpec]:
    """Assign ``n`` documents to folders proportionally to their weight.

    :param folders: the tree's folders.
    :param n: number of documents to place.
    :returns: one folder per document, grouped by folder (write order).
    """
    total_weight = sum(folder.weight for folder in folders)
    placed: list[FolderSpec] = []
    for folder in folders:
        count = round(n * folder.weight / total_weight)
        placed.extend([folder] * max(1, count))
    while len(placed) > n:
        placed.pop()
    index = 0
    while len(placed) < n:
        placed.append(folders[index % len(folders)])
        index += 1
    return placed


def _modified_at(folder_year: int | None, rng: random.Random, today: datetime) -> datetime:
    """Draw a modification time consistent with the folder's nominal year.

    The registry stores this as the document's ``reference_date``, which the
    compliance check reads as its age: an archive folder must really be old, or
    the retention check has nothing to work on.

    :param folder_year: nominal year, or ``None`` for a recent document.
    :param rng: seeded RNG.
    :param today: reference date (documents are never stamped in the future).
    :returns: the timestamp to apply to the file.
    """
    if folder_year is None:
        return today - timedelta(days=rng.randint(1, 540), hours=rng.randint(0, 23))
    stamp = datetime(folder_year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=rng.randint(0, 364), hours=rng.randint(8, 18)
    )
    return min(stamp, today - timedelta(days=1))


def _file_name(kind: str, index: int, rng: random.Random, long_name: bool) -> str:
    """Build a plausible file name for a document of a given archetype."""
    if long_name:
        return noise_module.LONG_NAME
    return f"{kind.replace('_', '-')}-{index:04d}" if rng.random() < 0.7 else (
        f"{kind.replace('_', '-')}_{rng.randint(1000, 9999)}"
    )


def plan_corpus(
    n: int,
    seed: int,
    *,
    profile: Profile = Profile.REALISTIC,
    formats: tuple[str, ...] = ("pdf", "docx", "txt"),
    with_noise: bool = True,
    ropa_file: Path | None = None,
    today: datetime | None = None,
) -> CorpusPlan:
    """Decide a whole corpus, without writing anything.

    :param n: number of real documents (noise is added on top).
    :param seed: reproducibility seed; the same seed yields an equal plan.
    :param profile: which tree to build (see
        :class:`~pii_detection.evaluation.enterprise.types.Profile`).
    :param formats: formats documents may be written in, intersected with what
        each folder allows.
    :param with_noise: also plant unsupported, pathological and awkwardly named
        files.
    :param ropa_file: register to build the ``ropa`` profile around; required
        for that profile.
    :param today: reference date for modification times (defaults to a fixed
        date, keeping plans reproducible).
    :returns: the planned corpus.
    :raises ValueError: if ``n`` is not positive, ``formats`` is empty, or the
        ``ropa`` profile is requested without a register file.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not formats:
        raise ValueError("at least one format is required")
    reference = today or _DEFAULT_TODAY
    rng = random.Random(seed)
    factory = PIIValueFactory(seed)

    rules: tuple[tuple[str, tuple[str, ...]], ...] = ()
    orphans: tuple[str, ...] = ()
    retention_expectations: tuple[RetentionExpectation, ...] = ()
    if profile is Profile.ROPA:
        if ropa_file is None:
            raise ValueError("the 'ropa' profile needs a register file (--ropa-file)")
        layout = ropa_layout(read_ropa(Path(ropa_file)))
        folders, rules, orphans = layout.folders, layout.rules, layout.orphan_types
        retention_expectations = layout.retention
    else:
        folders = REALISTIC_FOLDERS

    sizes = _size_sequence(n, rng)
    placement = _folder_sequence(folders, n)
    pii_free_budget = round(n * PII_FREE_SHARE)
    long_name_at = rng.randrange(n)

    documents: list[DocumentSpec] = []
    for index, (folder, size_class) in enumerate(zip(placement, sizes, strict=True)):
        kind = _pick_kind(folder, rng, pii_free_budget, len(documents), n)
        if not ARCHETYPES[kind].has_pii:
            pii_free_budget -= 1
        allowed = tuple(f for f in folder.formats if f in formats) or (formats[0],)
        file_format = rng.choice(allowed)
        body = build_body(ARCHETYPES[kind], factory, target_lines(size_class, rng))
        name = _file_name(kind, index + 1, rng, long_name=index == long_name_at)
        documents.append(
            DocumentSpec(
                relative_path=f"{folder.path}/{name}.{file_format}",
                kind=kind,
                size_class=size_class,
                file_format=file_format,
                annotated_text=body,
                modified_at=_modified_at(folder.year, rng, reference),
                expectation=Expectation.SCANNABLE,
            )
        )

    if with_noise:
        years = {folder.path: folder.year for folder in folders}
        documents.extend(
            noise_module.noise_specs(
                tuple(folder.path for folder in folders),
                rng,
                lambda path: _modified_at(years.get(path), rng, reference),
            )
        )
    return CorpusPlan(
        profile=profile,
        seed=seed,
        documents=tuple(documents),
        folder_rules=rules,
        expected_orphans=orphans,
        expected_retention=retention_expectations,
    )


def _pick_kind(
    folder: FolderSpec, rng: random.Random, pii_free_left: int, placed: int, total: int
) -> str:
    """Choose an archetype for a folder, steering towards the PII-free quota.

    The folder decides *what* may live in it; this only picks among those, with
    a nudge so the corpus lands near :data:`PII_FREE_SHARE` documents without
    PII — the half that measures false positives.

    :param folder: the folder being filled.
    :param rng: seeded RNG.
    :param pii_free_left: how many PII-free documents are still owed.
    :param placed: how many documents have been planned so far.
    :param total: how many documents the corpus has in total.
    :returns: the chosen archetype id.
    """
    free = tuple(k for k in folder.kinds if not ARCHETYPES[k].has_pii)
    bearing = tuple(k for k in folder.kinds if ARCHETYPES[k].has_pii)
    if not free:
        return rng.choice(bearing)
    if not bearing:
        return rng.choice(free)
    remaining = max(total - placed, 1)
    if rng.random() < min(1.0, max(0.0, pii_free_left / remaining)):
        return rng.choice(free)
    return rng.choice(bearing)


def write_corpus(plan: CorpusPlan, out_dir: Path) -> WriteSummary:
    """Materialise a plan: the tree, the gold, the manifest and the sources.

    Any previous ``tree/`` is **removed first**: writing a smaller (or
    differently seeded) corpus on top of an older one would leave files behind
    that no gold and no manifest describe — a corpus that lies about itself, and
    the surest way to spend an afternoon on a scan result that made no sense.

    :param plan: the corpus to write.
    :param out_dir: destination root (created if missing); ``tree/``,
        ``gold.jsonl``, ``manifest.jsonl`` and ``sources.jsonl`` are written
        under it.
    :returns: a summary of what was written.
    """
    out_dir = Path(out_dir)
    tree = out_dir / "tree"
    shutil.rmtree(tree, ignore_errors=True)
    shutil.rmtree(out_dir / "source", ignore_errors=True)  # layout before sources.jsonl
    tree.mkdir(parents=True, exist_ok=True)

    by_format: dict[str, int] = {}
    by_size: dict[str, int] = {}
    pii_free = 0
    total_bytes = 0
    with (out_dir / "gold.jsonl").open("w", encoding="utf-8") as gold_file, (
        out_dir / "manifest.jsonl"
    ).open("w", encoding="utf-8") as manifest_file, (
        out_dir / "sources.jsonl"
    ).open("w", encoding="utf-8") as sources_file:
        for spec in plan.documents:
            path = tree / spec.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if noise_module.is_noise(spec.kind):
                path.write_bytes(noise_module.payload_for(spec.kind))
            else:
                clean = parse_annotated_text(spec.relative_path, spec.annotated_text).text
                _write_document(clean, spec.file_format, path)
                sources_file.write(
                    json.dumps(
                        {"document_id": spec.relative_path, "annotated": spec.annotated_text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                by_size[spec.size_class] = by_size.get(spec.size_class, 0) + 1
            record = gold_record(spec.relative_path, spec.annotated_text)
            if spec.expectation is Expectation.SCANNABLE:
                gold_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                if not record["pii"]:
                    pii_free += 1
            manifest_file.write(json.dumps(_manifest_record(spec), ensure_ascii=False) + "\n")
            stamp = spec.modified_at.timestamp()
            os.utime(path, (stamp, stamp))
            by_format[spec.file_format] = by_format.get(spec.file_format, 0) + 1
            total_bytes += path.stat().st_size

    if plan.folder_rules:
        (out_dir / "folder_rules.json").write_text(
            json.dumps(
                [{"prefix": prefix, "activity_ids": list(ids)} for prefix, ids in
                 plan.folder_rules],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "expected_violations.json").write_text(
            json.dumps(
                {
                    "orphan_pii_types": list(plan.expected_orphans),
                    "retention_overdue": [
                        {
                            "prefix": expectation.prefix,
                            "activity_id": expectation.activity_id,
                            "retention_months": expectation.retention_months,
                            "age_months": expectation.age_months,
                        }
                        for expectation in plan.expected_retention
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return WriteSummary(
        root=out_dir,
        tree=tree,
        written=len(plan.documents),
        by_format=by_format,
        by_size=by_size,
        pii_free=pii_free,
        total_bytes=total_bytes,
    )


def _write_document(clean: str, file_format: str, path: Path) -> None:
    """Write one document in the requested format, reusing the renderers.

    :param clean: marker-free text, exactly what the file must contain.
    :param file_format: ``"pdf"``, ``"docx"`` or ``"txt"``.
    :param path: destination path.
    :raises ValueError: on an unknown format.
    """
    if file_format == "pdf":
        render_pdf(clean, path)
    elif file_format == "docx":
        render_docx(clean, path)
    elif file_format == "txt":
        path.write_text(clean, encoding="utf-8")
    else:
        raise ValueError(f"unsupported document format: {file_format!r}")


def _manifest_record(spec: DocumentSpec) -> dict[str, object]:
    """Build the manifest line of one planned file."""
    return {
        "document_id": spec.relative_path,
        "kind": spec.kind,
        "size_class": str(spec.size_class),
        "format": spec.file_format,
        "modified_at": spec.modified_at.isoformat(),
        "expectation": str(spec.expectation),
    }


def load_manifest(path: Path) -> list[dict[str, object]]:
    """Read a ``manifest.jsonl`` back.

    :param path: path to the manifest written by :func:`write_corpus`.
    :returns: the records, in write order.
    """
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def load_sources(path: Path) -> dict[str, str]:
    """Read a ``sources.jsonl`` back into ``{document_id: annotated_text}``.

    The annotated text is what value-level scoring needs (the clean text and the
    gold spans are both derived from it by
    :func:`~pii_detection.evaluation.corpus.parse_annotated_text`).

    :param path: path to the sources file written by :func:`write_corpus`.
    :returns: the annotated text of each document, keyed by id.
    """
    sources: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        sources[record["document_id"]] = record["annotated"]
    return sources


__all__ = [
    "PII_FREE_SHARE",
    "SIZE_MIX",
    "WriteSummary",
    "load_manifest",
    "load_sources",
    "plan_corpus",
    "write_corpus",
]
