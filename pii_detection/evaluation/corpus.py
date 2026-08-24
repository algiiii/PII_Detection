"""Annotated evaluation corpus for the detection layer.

Loads small hand-written documents where each PII is wrapped inline as
``{{pii_type:value}}``. The loader strips the markers, producing both the clean
text the detectors run on and the ground-truth spans in clean-text
coordinates — so the author never computes character offsets by hand.

This corpus is the yardstick to measure a detector's recall/precision (block
B4, Step 10). It is plain synthetic text: it does **not** need the document
ingestion layer (B3).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# {{pii_type:value}} — pii_type has no ':' or '}'; value is anything up to '}}'.
_ANNOTATION = re.compile(r"\{\{([^:}]+):(.*?)\}\}", re.DOTALL)


@dataclass(frozen=True)
class GroundTruthSpan:
    """Expected PII occurrence, in clean-text coordinates.

    :ivar start: start offset (inclusive) in the clean text.
    :ivar end: end offset (exclusive).
    :ivar pii_type: expected category, e.g. ``"iban"``.
    """

    start: int
    end: int
    pii_type: str


@dataclass(frozen=True)
class AnnotatedDocument:
    """A corpus document: clean text plus its ground-truth spans.

    :ivar document_id: stable identifier (the file stem, for file-backed docs).
    :ivar text: clean text, with every annotation marker removed.
    :ivar spans: expected PII occurrences, in declaration order.
    """

    document_id: str
    text: str
    spans: tuple[GroundTruthSpan, ...]


def parse_annotated_text(document_id: str, annotated: str) -> AnnotatedDocument:
    """Parse an inline-annotated string into clean text + ground-truth spans.

    Each ``{{pii_type:value}}`` marker is replaced by ``value`` in the clean
    text, and a :class:`GroundTruthSpan` is recorded over the interval that
    ``value`` ends up occupying. Text outside the markers is preserved verbatim.

    :param document_id: identifier to assign to the document.
    :param annotated: text with inline ``{{pii_type:value}}`` markers.
    :returns: the parsed :class:`AnnotatedDocument`.
    """
    parts: list[str] = []
    spans: list[GroundTruthSpan] = []
    clean_len = 0
    pos = 0
    for match in _ANNOTATION.finditer(annotated):
        before = annotated[pos : match.start()]
        parts.append(before)
        clean_len += len(before)

        pii_type = match.group(1).strip()
        value = match.group(2)
        parts.append(value)
        spans.append(GroundTruthSpan(clean_len, clean_len + len(value), pii_type))
        clean_len += len(value)
        pos = match.end()

    parts.append(annotated[pos:])
    return AnnotatedDocument(document_id, "".join(parts), tuple(spans))


def default_corpus_dir() -> Path:
    """Locate the annotated documents shipped with the package.

    :returns: absolute path to ``pii_detection/evaluation/documents``.
    """
    return Path(__file__).resolve().parent / "documents"


def load_corpus_dir(directory: Path | None = None) -> list[AnnotatedDocument]:
    """Load and parse every ``*.txt`` document in a directory.

    :param directory: folder of inline-annotated ``.txt`` files; defaults to the
        packaged :func:`default_corpus_dir`.
    :returns: the parsed documents, sorted by file name; the file stem becomes
        the ``document_id``.
    :raises FileNotFoundError: if the directory does not exist.
    """
    base = directory if directory is not None else default_corpus_dir()
    if not base.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {base}")
    return [
        parse_annotated_text(path.stem, path.read_text(encoding="utf-8"))
        for path in sorted(base.glob("*.txt"))
    ]


def load_corpus_jsonl(path: Path) -> list[AnnotatedDocument]:
    """Load an annotated corpus from a JSON Lines file.

    Each line is an object with a ``document_id`` and an ``annotated`` field
    holding the inline-marked text — the ``sources.jsonl`` emitted next to the
    enterprise corpus tree. Parsing it here means the enterprise corpus, built
    for the folder-scale stress test, can also feed the detector benchmarks
    without a second annotation format.

    :param path: the ``.jsonl`` file to read.
    :returns: the parsed documents, in file order.
    :raises FileNotFoundError: if the file does not exist.
    :raises ValueError: if a line is not valid JSON or lacks the expected fields.
    """
    if not path.is_file():
        raise FileNotFoundError(f"corpus file not found: {path}")
    documents: list[AnnotatedDocument] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            document_id = str(record["document_id"])
            annotated = str(record["annotated"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{number}: malformed corpus record ({exc})") from exc
        documents.append(parse_annotated_text(document_id, annotated))
    return documents


def load_annotated_corpus(source: Path | None = None) -> list[AnnotatedDocument]:
    """Load an annotated corpus from either layout, picked by what ``source`` is.

    The single entry point the evaluation runners call, so a corpus can be
    swapped on the command line regardless of how it is stored: a **directory**
    of ``.txt`` files (:func:`load_corpus_dir`) or a **JSON Lines** file
    (:func:`load_corpus_jsonl`).

    :param source: directory or ``.jsonl`` file; defaults to the packaged corpus.
    :returns: the parsed documents.
    :raises FileNotFoundError: if ``source`` is neither an existing directory nor
        an existing file.
    """
    if source is None:
        return load_corpus_dir(None)
    if source.is_dir():
        return load_corpus_dir(source)
    if source.is_file():
        return load_corpus_jsonl(source)
    raise FileNotFoundError(f"corpus not found: {source}")


__all__ = [
    "GroundTruthSpan",
    "AnnotatedDocument",
    "parse_annotated_text",
    "default_corpus_dir",
    "load_corpus_dir",
    "load_corpus_jsonl",
    "load_annotated_corpus",
]
