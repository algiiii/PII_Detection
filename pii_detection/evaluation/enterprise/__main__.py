"""CLI of the enterprise corpus generator::

    python -m pii_detection.evaluation.enterprise --out corpus/generated --n 300

The output root holds ``tree/`` — **the only thing to point a scan at** — plus
the gold, the manifest and the annotated sources beside it, all as ``.jsonl``
files no document reader supports, so pointing the scan at the root by mistake
still only ingests the corpus. With
``--profile ropa`` the tree is built around a real register file and two extra
artefacts appear: the folder rules to load, and the violations the compliance
check is expected to report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pii_detection.evaluation.enterprise.builder import plan_corpus, write_corpus
from pii_detection.evaluation.enterprise.types import Profile

#: Default output root: disposable, regenerable, and already git-ignored.
DEFAULT_OUT = Path("corpus/generated")

#: Register used by the ``ropa`` profile when none is given.
DEFAULT_ROPA = Path("corpus/ropa/ropa_aziendale.ods")


def main(argv: list[str] | None = None) -> None:
    """Generate a corpus and write it to disk.

    :param argv: argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        description="Generate a synthetic enterprise document tree for stress testing."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=300, help="number of documents")
    parser.add_argument("--seed", type=int, default=42, help="reproducibility seed")
    parser.add_argument(
        "--profile",
        choices=[str(p) for p in Profile],
        default=str(Profile.REALISTIC),
        help="realistic: a plain company share; ropa: built around a register",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["pdf", "docx", "txt"],
        default=["pdf", "docx", "txt"],
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="skip unsupported, pathological and awkwardly named files",
    )
    parser.add_argument(
        "--ropa-file",
        type=Path,
        default=None,
        help=f"register for the ropa profile (default: {DEFAULT_ROPA})",
    )
    args = parser.parse_args(argv)

    profile = Profile(args.profile)
    ropa_file = args.ropa_file or (DEFAULT_ROPA if profile is Profile.ROPA else None)
    plan = plan_corpus(
        args.n,
        args.seed,
        profile=profile,
        formats=tuple(args.formats),
        with_noise=not args.no_noise,
        ropa_file=ropa_file,
    )
    summary = write_corpus(plan, args.out)

    scannable = len(plan.scannable())
    print(f"wrote {summary.written} files ({summary.total_bytes / 1_000_000:.1f} MB) "
          f"to {summary.tree}")
    print("  formats: " + ", ".join(f"{k}={v}" for k, v in sorted(summary.by_format.items())))
    print("  lengths: " + ", ".join(f"{k}={v}" for k, v in sorted(summary.by_size.items())))
    print(f"  documents to scan: {scannable}, of which without PII: {summary.pii_free} "
          f"({summary.pii_free / scannable:.0%})")
    print(f"  noise: {summary.written - args.n} files (skipped/error/empty)")
    print(f"  gold: {summary.root / 'gold.jsonl'} — manifest: {summary.root / 'manifest.jsonl'}")
    # Spelled out because pointing a scan one level too high is the easy mistake
    # to make, and it silently ingests the wrong thing.
    print(f"\nscan THIS folder (and only this one): {summary.tree}")
    if plan.folder_rules:
        print(f"  folder rules: {summary.root / 'folder_rules.json'} "
              f"({len(plan.folder_rules)} prefixes)")
        print(f"  expected orphan types: {', '.join(plan.expected_orphans) or 'none'}")


if __name__ == "__main__":
    main()


__all__ = ["main"]
