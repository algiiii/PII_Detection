"""Tests for the synthetic corpus generator.

Certify the corpus without touching Presidio: values pass their own checksums,
generation is reproducible, and the whole ``pii_type`` catalog is exercised. The
module needs the ``[eval]`` optional deps, so the whole file is skipped when they
are absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("faker")
pytest.importorskip("codicefiscale")

from pathlib import Path  # noqa: E402

from pii_detection.evaluation.corpus import load_corpus_dir, parse_annotated_text  # noqa: E402
from pii_detection.evaluation.corpus_generator import (  # noqa: E402
    generate_documents,
    validate_factory,
    write_corpus,
)

_CATALOG = {
    "email", "phone", "iban", "credit_card", "swiss_avs", "ip_address",
    "person_name", "address", "date_of_birth", "italian_id", "health_data",
}


def test_generated_values_pass_their_checksums() -> None:
    """IBAN/credit card/AVS/codice fiscale must satisfy their own checksums, so
    the generator never becomes the reason a checksum-validating detector misses."""
    validate_factory(seed=7, rounds=100)


def test_generation_is_reproducible_for_a_seed() -> None:
    """Same seed -> byte-identical corpus (needed to compare runs)."""
    assert generate_documents(12, seed=1) == generate_documents(12, seed=1)


def test_corpus_covers_the_whole_catalog() -> None:
    """Every catalog category appears at least once across the templates."""
    seen: set[str] = set()
    for doc_id, text in generate_documents(len(_CATALOG) * 6, seed=5):
        seen.update(span.pii_type for span in parse_annotated_text(doc_id, text).spans)
    assert _CATALOG <= seen


def test_annotations_round_trip_through_the_loader() -> None:
    """The emitted markers parse back into non-empty, catalog-typed gold spans."""
    (doc_id, text), *_ = generate_documents(1, seed=3)
    parsed = parse_annotated_text(doc_id, text)
    assert parsed.spans
    for span in parsed.spans:
        assert span.pii_type in _CATALOG
        assert parsed.text[span.start : span.end]


def test_emit_clean_writes_derived_files_the_loader_ignores(tmp_path: Path) -> None:
    """--emit-clean writes marker-free copies under clean/ that the corpus loader
    does not pick up (its glob is non-recursive), so no gold-zero duplicates leak."""
    written = write_corpus(tmp_path, n=6, seed=2, emit_clean=True)
    clean_files = sorted((tmp_path / "clean").glob("*.txt"))

    assert len(clean_files) == len(written)
    assert all("{{" not in p.read_text(encoding="utf-8") for p in clean_files)
    # The loader sees only the annotated sources, not the derived clean copies.
    assert len(load_corpus_dir(tmp_path)) == len(written)
