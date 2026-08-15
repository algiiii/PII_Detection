"""Did the scan find what the corpus planted? Checked against the registry.

Run after scanning a generated corpus::

    python -m pii_detection.evaluation.enterprise.verify corpus/generated

Two things are checked, and one deliberately is not.

**Detection, by type count.** The registry stores only references — type,
position, provenance — and never the PII *value* (minimization, block B5). So
the comparison with the gold can only be a multiset comparison of ``pii_type``
counts per document: one detected ``email`` matches one expected ``email``, and
which email it was is a question the registry cannot answer by design. The
counts feed the same :class:`~pii_detection.evaluation.scoring.Metrics` and the
same table as the other scorers. Value-level scoring stays available on the
annotated text kept under ``source/``, via
:func:`~pii_detection.evaluation.scoring.evaluate_values`.

**Handling, by expectation.** Every planted file declared in the manifest what
the scan had to do with it: a ``scannable`` file must be in the registry, a
``skipped`` or ``error`` one must not. Any mismatch is reported by name.

The numbers are only as good as the detectors that produced them: with the
lightweight local stack ``health_data`` and most names are simply not detectable
(they need GLiNER), so a run outside the container will show recall gaps that
say something about the deployment, not about the corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pii_detection.evaluation.enterprise.builder import load_manifest
from pii_detection.evaluation.render import load_gold
from pii_detection.evaluation.scoring import (
    EvaluationReport,
    format_report,
    report_from_counters,
)
from pii_detection.registry.repository import PIIRepository


@dataclass(frozen=True)
class HandlingReport:
    """How the scan treated the planted files, against their declared fate.

    :ivar expected: number of files per expectation, as planned.
    :ivar missing: ids of ``scannable`` files absent from the registry — the
        scan failed to ingest something it should have.
    :ivar unexpected: ids of ``skipped``/``error`` files present in the registry
        — the scan ingested something it should have refused.
    """

    expected: dict[str, int]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def clean(self) -> bool:
        """:returns: whether every file was handled as the corpus declared."""
        return not self.missing and not self.unexpected


def compare_types(
    gold: dict[str, list[tuple[str, str]]], repository: PIIRepository
) -> EvaluationReport:
    """Score the registry against the gold, counting types per document.

    :param gold: the corpus gold, as returned by
        :func:`~pii_detection.evaluation.render.load_gold`.
    :param repository: the registry the scan wrote into.
    :returns: the per-category report.
    """
    tp_by: Counter[str] = Counter()
    fp_by: Counter[str] = Counter()
    fn_by: Counter[str] = Counter()
    for document_id, expected in gold.items():
        wanted = Counter(pii_type for pii_type, _ in expected)
        found = Counter(
            instance.pii_type for instance in repository.instances_for(document_id)
        )
        for pii_type in set(wanted) | set(found):
            matched = min(wanted[pii_type], found[pii_type])
            tp_by[pii_type] += matched
            fp_by[pii_type] += found[pii_type] - matched
            fn_by[pii_type] += wanted[pii_type] - matched
    return report_from_counters(tp_by, fp_by, fn_by)


def check_handling(
    manifest: list[dict[str, object]], repository: PIIRepository
) -> HandlingReport:
    """Check each planted file ended up where its expectation said it would.

    :param manifest: records from :func:`~...builder.load_manifest`.
    :param repository: the registry the scan wrote into.
    :returns: the handling report.
    """
    recorded = {document.document_id for document in repository.documents()}
    expected: Counter[str] = Counter()
    missing: list[str] = []
    unexpected: list[str] = []
    for record in manifest:
        document_id = str(record["document_id"])
        expectation = str(record["expectation"])
        expected[expectation] += 1
        if expectation == "scannable" and document_id not in recorded:
            missing.append(document_id)
        elif expectation != "scannable" and document_id in recorded:
            unexpected.append(document_id)
    return HandlingReport(dict(expected), tuple(missing), tuple(unexpected))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: score a scanned corpus against its own gold.

    :param argv: argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        description="Compare the PII registry with the gold of a generated corpus."
    )
    parser.add_argument("root", type=Path, help="corpus root (holds gold.jsonl)")
    parser.add_argument("--db-url", default=None, help="registry database URL")
    args = parser.parse_args(argv)

    repository = PIIRepository(args.db_url) if args.db_url else PIIRepository()
    gold = load_gold(args.root / "gold.jsonl")
    print(format_report(compare_types(gold, repository)))

    handling = check_handling(load_manifest(args.root / "manifest.jsonl"), repository)
    print()
    print("planned files: " + ", ".join(f"{k}={v}" for k, v in sorted(handling.expected.items())))
    if handling.clean:
        print("handling: every file was treated as declared")
    else:
        for document_id in handling.missing:
            print(f"MISSING (should have been scanned): {document_id}")
        for document_id in handling.unexpected:
            print(f"UNEXPECTED (should not be in the registry): {document_id}")


if __name__ == "__main__":
    main()


__all__ = ["HandlingReport", "check_handling", "compare_types", "main"]
