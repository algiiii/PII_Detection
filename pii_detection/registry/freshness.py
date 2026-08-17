"""File stamp: has this file changed since the registry last looked at it? (B5)

The *technical* counterpart of
:class:`~pii_detection.extraction.dates.ReferenceDate`, and the two must not be
confused. The reference date is **semantic** — how old the content is, possibly
read from inside the file — and answers the retention question of block B7. A
:class:`FileStamp` is **technical** — the observable state of the file on disk —
and answers a different one: is it worth extracting and analysing this document
again?

Using the semantic date for that second question would be a silent bug: a PDF's
internal ``modDate`` frequently stays put when the file is rewritten, so a document
edited yesterday would look untouched and be skipped forever. Hence two separate
notions, recorded side by side on the document.

The stamp is deliberately the cheapest signal that works — modification time and
size, one ``stat`` call — rather than a content hash: hashing every file of a large
share would cost exactly what the incremental scan is meant to save.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class FileStamp:
    """The observable state of a file, as the registry recorded it.

    :ivar modified_at: the file system modification time, timezone-aware (UTC).
    :ivar size: the file size in bytes.
    """

    modified_at: datetime
    size: int


def stamp_for(path: str | Path) -> FileStamp:
    """Read a file's current stamp.

    :param path: path to an existing file.
    :returns: its :class:`FileStamp`.
    :raises OSError: if the file cannot be stat'ed (missing, unreadable).
    """
    status = Path(path).stat()
    return FileStamp(
        modified_at=datetime.fromtimestamp(status.st_mtime, tz=timezone.utc),
        size=status.st_size,
    )


def detector_signature(
    detector_ids: Iterable[str], *, config_dir: Path | None = None
) -> str:
    """Fingerprint the detection engine a scan is about to run with.

    Skipping unchanged files is only safe as long as the *engine* is unchanged
    too: enabling GLiNER or adding a rule to ``custom_patterns.yaml`` changes what
    would be found in a file whose bytes never moved, and a registry that keeps
    showing the old results while claiming to be up to date is worse than one that
    is merely stale — it lies. Recording this signature next to the file stamp lets
    a configuration change invalidate the skip on its own, without the operator
    having to remember ``--full``.

    :param detector_ids: identifiers of the detectors taking part in the scan.
    :param config_dir: directory holding the YAML detection configuration;
        defaults to the one shipped with the package.
    :returns: a short hexadecimal digest of the detector ids and the configuration
        contents.
    """
    from pii_detection.detection.config import default_config_dir  # lazy: config layer

    digest = hashlib.sha256()
    for detector_id in sorted(detector_ids):
        digest.update(detector_id.encode())
        digest.update(b"\0")
    directory = config_dir if config_dir is not None else default_config_dir()
    for path in sorted(directory.glob("*.yaml")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def needs_rescan(
    current: FileStamp,
    recorded: FileStamp | None,
    *,
    signature: str | None = None,
    recorded_signature: str | None = None,
    tolerance_seconds: float = 1.0,
) -> bool:
    """Whether a file must be analysed again.

    The rule compares *difference*, not recency: a file restored to an older
    version has changed just as much as one that was edited, and a "newer than"
    test would skip it forever. The tolerance absorbs the differing timestamp
    granularity of file systems (bind mounts, FAT, container layers), which would
    otherwise make every file look modified.

    :param current: the file's stamp right now.
    :param recorded: the stamp the registry holds, or ``None`` if the document was
        never scanned — in which case it obviously must be.
    :param signature: fingerprint of the engine about to run, from
        :func:`detector_signature`; omit to ignore engine changes.
    :param recorded_signature: the fingerprint stored with the last scan.
    :param tolerance_seconds: how far the two modification times may differ before
        the file counts as changed.
    :returns: ``True`` when the document must be analysed again.
    """
    if recorded is None:
        return True
    if signature is not None and signature != recorded_signature:
        return True
    if current.size != recorded.size:
        return True
    drift = abs((current.modified_at - recorded.modified_at).total_seconds())
    return drift > tolerance_seconds


__all__ = ["FileStamp", "stamp_for", "detector_signature", "needs_rescan"]
