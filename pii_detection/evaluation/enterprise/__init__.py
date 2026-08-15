"""Synthetic *enterprise* corpus generator: a whole file share, from a seed.

Where :mod:`pii_detection.evaluation.corpus_generator` produces a flat batch of
short documents to measure the detectors on a single file, this package produces
what the system actually meets in production: a **folder tree** of hundreds of
documents of realistic shape, length and messiness, so the recursive scan
(:mod:`pii_detection.registry.scan_folder`), the path-based identity, the folder
rules (:mod:`pii_detection.registry.folder_rules`), the compliance verdict and
the dashboard can all be put under load.

Three properties make the corpus useful rather than merely big:

* **ground truth for free** — every document is generated as annotated text
  (``{{pii_type:value}}``) and written out through
  :func:`~pii_detection.evaluation.render.gold_record`, so a ``gold.jsonl``
  keyed by the very same ``document_id`` the scan computes comes out of the same
  pass;
* **documents without PII** — roughly a third of the tree is minutes, policies,
  manuals and price lists sprinkled with *distractors* (protocol numbers,
  amounts, product codes). They are the only way to measure false positives;
* **declared messiness** — unsupported formats, pathological files and hostile
  names are planted on purpose and recorded in the manifest with the outcome
  they must produce (skipped, or an isolated error), so a test asserts the
  system behaves *exactly* right instead of merely surviving.

The corpus plan is a pure value (:func:`plan_corpus`); writing it to disk is a
separate step. Requires ``[eval]`` (Faker, codicefiscale, fpdf2) and
``[extraction]`` (python-docx) — no new dependency.
"""

from __future__ import annotations

from pii_detection.evaluation.enterprise.types import (
    CorpusPlan,
    DocumentSpec,
    Expectation,
    FolderSpec,
    Profile,
    SizeClass,
)

__all__ = [
    "CorpusPlan",
    "DocumentSpec",
    "Expectation",
    "FolderSpec",
    "Profile",
    "SizeClass",
]
