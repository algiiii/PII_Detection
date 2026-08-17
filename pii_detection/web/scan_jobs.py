"""In-memory background jobs for folder scans (block B8).

The web app must not block on a long GLiNER scan, so a scan runs in a worker
thread and the UI polls its status. Jobs live in an **in-memory** registry — valid
for the single-worker uvicorn the app runs under; a process restart forgets them
(acceptable for this operator tool — job persistence is out of scope).
"""

from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import dataclass

from pii_detection.detection.protocol import PIIDetector
from pii_detection.registry.folder_rules import ApplyRulesResult
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.scan_folder import FolderScanResult, ingest_folder


@dataclass
class ScanJob:
    """State of a background folder scan.

    :ivar id: unique job id.
    :ivar folder: the folder being scanned.
    :ivar use_gliner: whether GLiNER is used for the NER.
    :ivar prune: reconcile documents gone from the folder as removed; ``False`` for
        uploads (a partial upload must not prune the rest of the registry).
    :ivar incremental: skip files unchanged since their last scan; ``False`` for
        uploads, whose files carry the upload time as their modification time and
        would therefore all look modified anyway.
    :ivar state: ``"running"``, ``"done"`` or ``"error"``.
    :ivar done: files processed so far.
    :ivar total: files to process.
    :ivar result: the summary once finished, else ``None``.
    :ivar rules_applied: folder-rule application summary once finished, else ``None``.
    :ivar error: the error message on failure, else ``None``.
    :ivar cleanup_dir: a temporary directory to remove once the job ends (set for
        uploads, whose files are materialized under it); ``None`` to keep the folder.
    """

    id: str
    folder: str
    use_gliner: bool
    prune: bool = True
    incremental: bool = True
    state: str = "running"
    done: int = 0
    total: int = 0
    result: FolderScanResult | None = None
    rules_applied: ApplyRulesResult | None = None
    error: str | None = None
    cleanup_dir: str | None = None


_JOBS: dict[str, ScanJob] = {}
_LOCK = threading.Lock()


def _build_detectors(use_gliner: bool) -> tuple[PIIDetector, PIIDetector]:
    """Build the default detectors (monkeypatched to fakes in tests).

    Lazy import so importing this module needs no Presidio stack.

    :param use_gliner: use GLiNER for the NER instead of spaCy.
    :returns: the ``(pattern, ner)`` detector pair.
    """
    from pii_detection.detection.presidio_detector import build_default_detectors

    return build_default_detectors(use_gliner=use_gliner)


def get_job(job_id: str) -> ScanJob | None:
    """:param job_id: id to look up. :returns: the job, or ``None`` if unknown."""
    with _LOCK:
        return _JOBS.get(job_id)


def start_scan_job(
    folder: str,
    *,
    use_gliner: bool,
    prune: bool = True,
    incremental: bool = True,
    cleanup_dir: str | None = None,
) -> str:
    """Start a folder scan in a background thread and return its job id.

    :param folder: path to scan (a server-side path, or a temp dir of uploaded files).
    :param use_gliner: use GLiNER for the NER.
    :param prune: reconcile documents gone from the folder as removed; pass ``False``
        for uploads so a partial upload does not prune the rest of the registry.
    :param incremental: skip files unchanged since their last scan; pass ``False``
        to force a full re-analysis.
    :param cleanup_dir: a temporary directory to delete when the job ends (uploads);
        ``None`` leaves the folder in place.
    :returns: the new job id, to poll via :func:`get_job`.
    """
    job = ScanJob(
        id=uuid.uuid4().hex,
        folder=folder,
        use_gliner=use_gliner,
        prune=prune,
        incremental=incremental,
        cleanup_dir=cleanup_dir,
    )
    with _LOCK:
        _JOBS[job.id] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return job.id


def _run(job: ScanJob) -> None:
    """Worker body: build detectors, scan the folder, record the outcome."""

    def _progress(done: int, total: int) -> None:
        with _LOCK:
            job.done, job.total = done, total

    try:
        pattern, ner = _build_detectors(job.use_gliner)
        repository = PIIRepository()
        result = ingest_folder(
            job.folder,
            pattern,
            ner,
            repository=repository,
            prune=job.prune,
            incremental=job.incremental,
            progress=_progress,
        )
        applied = repository.apply_folder_rules()
        with _LOCK:
            job.result = result
            job.rules_applied = applied
            job.state = "done"
    except Exception as exc:  # noqa: BLE001 — surface any failure on the status page
        with _LOCK:
            job.error = str(exc)
            job.state = "error"
    finally:
        if job.cleanup_dir is not None:
            shutil.rmtree(job.cleanup_dir, ignore_errors=True)


__all__ = ["ScanJob", "get_job", "start_scan_job"]
