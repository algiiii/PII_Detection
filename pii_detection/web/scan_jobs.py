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
from collections import Counter
from dataclasses import dataclass

from pii_detection.detection.ai_detector import AITriggerPolicy
from pii_detection.detection.protocol import PIIDetector
from pii_detection.registry.folder_rules import ApplyRulesResult
from pii_detection.registry.freshness import detector_signature
from pii_detection.registry.ingest import ingest_document
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.scan_folder import FolderScanResult, ingest_folder


@dataclass
class ScanJob:
    """State of a background folder scan.

    :ivar id: unique job id.
    :ivar folder: the folder being scanned.
    :ivar use_gliner: whether GLiNER is used for the NER.
    :ivar ai_rate: the AI sampling knob for this scan — ``0`` no AI, ``1`` every
        document, ``N`` roughly one in ``N`` (see
        :class:`~pii_detection.detection.ai_detector.AITriggerPolicy`).
    :ivar document_id: set when this is a **single-document** AI re-analysis (the
        per-document trigger); ``None`` for a folder scan. When set, :ivar:`folder`
        holds that document's file path.
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
    ai_rate: int = 0
    document_id: str | None = None
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


def _build_detectors(
    use_gliner: bool, ai_rate: int
) -> tuple[PIIDetector, PIIDetector, PIIDetector | None]:
    """Build the detectors for a scan (monkeypatched to fakes in tests).

    Lazy import so importing this module needs no Presidio nor Ollama stack. The AI
    detector is built only when the scan uses it (``ai_rate > 0``); otherwise it is
    ``None`` and no AI runs.

    :param use_gliner: use GLiNER for the NER instead of spaCy.
    :param ai_rate: the AI sampling knob (``0`` no AI, ``1`` all, ``N`` one-in-``N``).
    :returns: the ``(pattern, ner, ai)`` triple; ``ai`` is ``None`` when no AI runs.
    """
    from pii_detection.detection.presidio_detector import build_default_detectors

    pattern, ner = build_default_detectors(use_gliner=use_gliner)
    ai: PIIDetector | None = None
    if ai_rate > 0:
        from pii_detection.detection.ai_detector import build_ai_detector

        ai = build_ai_detector()
    return pattern, ner, ai


def get_job(job_id: str) -> ScanJob | None:
    """:param job_id: id to look up. :returns: the job, or ``None`` if unknown."""
    with _LOCK:
        return _JOBS.get(job_id)


def active_jobs() -> list[ScanJob]:
    """List the jobs still running, for the "a scan is in progress" indicator.

    Read by the shared page header (a Jinja global) so any page shows that a scan is
    active without every route having to pass it.

    :returns: the jobs in state ``"running"``.
    """
    with _LOCK:
        return [job for job in _JOBS.values() if job.state == "running"]


def start_scan_job(
    folder: str,
    *,
    use_gliner: bool,
    ai_rate: int = 0,
    prune: bool = True,
    incremental: bool = True,
    cleanup_dir: str | None = None,
) -> str:
    """Start a folder scan in a background thread and return its job id.

    :param folder: path to scan (a server-side path, or a temp dir of uploaded files).
    :param use_gliner: use GLiNER for the NER.
    :param ai_rate: AI sampling knob (``0`` no AI, ``1`` all, ``N`` one-in-``N``).
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
        ai_rate=ai_rate,
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
        pattern, ner, ai = _build_detectors(job.use_gliner, job.ai_rate)
        repository = PIIRepository()
        # The rate drives which documents get the AI pass (0 none, 1 all, N one-in-N).
        policy = AITriggerPolicy(sampling_rate=job.ai_rate)
        result = ingest_folder(
            job.folder,
            pattern,
            ner,
            ai,
            repository=repository,
            prune=job.prune,
            incremental=job.incremental,
            progress=_progress,
            ai_policy=policy,
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


def start_document_ai_job(document_id: str, path: str) -> str:
    """Re-analyse a single document with the AI in the background, return the job id.

    The per-document on-demand trigger. The AI pass on a long document with a large
    model on CPU takes minutes, so it must not run inside the request: this reuses the
    same :class:`ScanJob` machinery (thread + in-memory registry + status polling) for
    one file. The whole pipeline (pattern + NER + AI) is re-run and recorded so the B5
    delta and the CONFIRMED refresh apply identically; only the outcome differs (the AI
    may now confirm or discover PII). The document text is re-extracted from ``path``
    and never persisted (minimization).

    :param document_id: identity of the document in the registry.
    :param path: the document's file path, to re-extract and re-scan.
    :returns: the new job id, to poll via :func:`get_job`.
    """
    job = ScanJob(
        id=uuid.uuid4().hex,
        folder=path,
        use_gliner=False,
        ai_rate=1,  # on-demand: always AI on this one document
        document_id=document_id,
    )
    with _LOCK:
        _JOBS[job.id] = job
    threading.Thread(target=_run_document, args=(job,), daemon=True).start()
    return job.id


def _run_document(job: ScanJob) -> None:
    """Worker body for the per-document AI re-analysis (one file, always with AI)."""
    try:
        pattern, ner, ai = _build_detectors(job.use_gliner, job.ai_rate)
        repository = PIIRepository()
        signature = detector_signature([pattern.detector_id, ner.detector_id])
        assert job.document_id is not None  # set by start_document_ai_job
        ingest_document(
            job.folder,
            pattern,
            ner,
            document_id=job.document_id,
            ai=ai,
            repository=repository,
            detector_signature=signature,
        )
        counts: Counter[str] = Counter()
        for instance in repository.instances_for(job.document_id):
            counts[instance.pii_type] += 1
        with _LOCK:
            job.done = job.total = 1
            job.result = FolderScanResult(scanned=1, ai_documents=1, by_type=dict(counts))
            job.state = "done"
    except Exception as exc:  # noqa: BLE001 — surface any failure on the status page
        with _LOCK:
            job.error = str(exc)
            job.state = "error"


__all__ = ["ScanJob", "get_job", "active_jobs", "start_scan_job", "start_document_ai_job"]
