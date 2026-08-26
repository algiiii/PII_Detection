"""Generative-AI detector — the sampled "second opinion" of block B4.

This is the AI technique the architecture always reserved a slot for
(:attr:`~pii_detection.detection.types.DetectorKind.AI`,
:attr:`~pii_detection.detection.types.ConfirmationLevel.AI_DISCOVERED`) but never
fed: a local, CPU-friendly LLM reads the document and reports the PII it finds,
as a **second opinion** next to the regex/NER detectors. Its purpose is the
thesis question — *when does generative AI, at a higher energy cost, beat the
traditional detectors, and when does it not?* — so it is a full
:class:`~pii_detection.detection.protocol.PIIDetector`, measurable on its own.

It mirrors the proven pattern of
:class:`~pii_detection.ropa.ingestion.category_mapper.LLMCategoryMapper`:

- a **closed catalog** in the prompt, so the model can only assign declared
  ``pii_type`` ids; every id it returns is re-validated against the catalog and
  unknown ones are dropped (an hallucination can never reach the pipeline);
- a **defensive JSON parse** (extract the array even from prose or ``<think>``
  blocks) and **anti-hallucination on values**: the LLM returns *values*, not
  offsets — offsets from an LLM are unreliable — so every value is located in the
  text with :meth:`str.find`; a value that does not occur verbatim is discarded;
- **graceful degradation**: the model runs on a possibly-unreachable local
  runtime, so any per-chunk failure is logged once and skipped, and an
  unreachable Ollama yields ``[]``. The AI is an enhancement, never a single
  point of failure.

The trigger is a **policy of the caller**, not of the detector:
:class:`AITriggerPolicy` decides *which* documents of a batch get the (costly)
AI pass; the detector itself always analyses the text it is given.

**Minimization (§2.3.11).** The chunk text (PII values included) leaves the
process only toward the *local* Ollama container, never a cloud API; the
``rationale`` may quote a value, so it lives on
:class:`~pii_detection.detection.types.DetectionProvenance` in memory but is
never persisted (see the B5 registry).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from pii_detection.detection.config import (
    PIICategoryCatalog,
    default_config_dir,
    load_category_catalog,
)
from pii_detection.detection.protocol import BaseDetector
from pii_detection.detection.types import DetectorKind, PIICandidate, TextSpan
from pii_detection.llm.client import LLMClient

#: Confidence of a single, unverified LLM claim: below a checksum-validated regex
#: hit, above nothing. The merge (Step 2) raises it when the AI confirms another
#: source.
DEFAULT_AI_CONFIDENCE = 0.6

#: Sliding-window size (characters) the document is chunked into for the LLM. A
#: chunk (~500–700 tokens) stays well under any default context window.
CHUNK_SIZE = 2000

#: Overlap (characters) between consecutive chunks, so a PII straddling a window
#: boundary is still seen whole by at least one chunk (>= any realistic PII).
CHUNK_OVERLAP = 200

#: Environment variable selecting the detector's model, kept separate from the
#: mapper's ``ROPA_LLM_MODEL`` so the two AI tasks can run different models.
PII_LLM_MODEL_ENV = "PII_LLM_MODEL"

#: Environment variable holding the 1-in-N sampling rate (``0``/absent = off).
PII_AI_SAMPLING_RATE_ENV = "PII_AI_SAMPLING_RATE"

#: Environment variable overriding the generated-token cap (:data:`DEFAULT_AI_NUM_PREDICT`).
PII_LLM_NUM_PREDICT_ENV = "PII_LLM_NUM_PREDICT"

#: Default cap on tokens the LLM may generate per chunk. The answer is a short
#: JSON list of PII values, so a few hundred tokens suffice; the cap stops a
#: rambling (e.g. reasoning) model from emitting thousands of chain-of-thought
#: tokens, which would dominate latency/energy and can break the JSON parse.
DEFAULT_AI_NUM_PREDICT = 1024

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)

_logger = logging.getLogger(__name__)

_AI_SYSTEM_PROMPT = (
    "You detect personal data (PII) in documents, as a second opinion alongside "
    "traditional detectors. The documents are written in Italian. Be conservative and "
    "literal: report only values that appear verbatim in the text, assign only ids from "
    "the catalog given by the user, and when in doubt do not report anything. Answer with "
    "JSON only, no prose."
)


class AIResponseParser:
    """Parse the model's JSON answer into validated ``(value, type, rationale)``.

    Reused by :class:`LLMDetector` for every chunk. It is defensive on both axes:
    the JSON array is extracted even when wrapped in prose or a ``<think>`` block
    (some models emit reasoning), and every ``pii_type`` is validated against the
    closed catalog so an id the catalog does not declare is silently dropped.

    :ivar _catalog: catalog every returned ``pii_type`` is validated against
        (private).
    """

    def __init__(self, catalog: PIICategoryCatalog) -> None:
        """Store the validating catalog.

        :param catalog: catalog every returned ``pii_type`` must belong to.
        """
        self._catalog = catalog

    def parse(self, answer: str) -> list[tuple[str, str, str | None]]:
        """Extract the JSON array and keep only catalog-valid, verbatim entries.

        :param answer: the raw text returned by the model.
        :returns: one ``(value, pii_type, rationale)`` triple per accepted entry,
            in order; entries with a non-string/empty ``value`` or a ``pii_type``
            absent from the catalog are dropped.
        :raises ValueError: if no JSON array can be found in the answer.
        """
        match = _JSON_ARRAY.search(answer)
        if match is None:
            raise ValueError("no JSON array in LLM answer")
        result: list[tuple[str, str, str | None]] = []
        for item in json.loads(match.group(0)):
            pii_type = item.get("pii_type")
            value = item.get("value")
            if pii_type not in self._catalog or not isinstance(value, str) or not value:
                continue
            rationale = item.get("rationale")
            result.append((value, str(pii_type), str(rationale) if rationale else None))
        return result


class LLMDetector(BaseDetector):
    """Detector backed by a local LLM, a second opinion over the whole document.

    Same :class:`~pii_detection.detection.protocol.PIIDetector` contract as the
    regex and NER detectors, so the merge and the benchmark treat it uniformly.
    :attr:`detector_id` embeds the model name (``"ai.<model>"``) so UI, registry
    and benchmark identify *which* model produced a detection.

    The document is scanned in overlapping windows (:data:`CHUNK_SIZE` /
    :data:`CHUNK_OVERLAP`): each chunk is sent to the model, the returned values
    are located back in the text, and the global spans are de-duplicated across
    the overlap regions. :meth:`detect` never raises — any per-chunk failure is
    logged once and skipped.

    :ivar _client: the shared LLM client (private).
    :ivar _parser: the response parser bound to the catalog (private).
    :ivar _confidence: confidence stamped on every candidate (private).
    :ivar _catalog: the closed catalog embedded in the prompt (private).
    :ivar chunks_seen: chunks submitted to the model since construction.
    :ivar chunks_failed: chunks whose analysis failed and was skipped. Skipping keeps
        a scan running, but a measurement taken while chunks are being dropped is not
        a measurement of the model: callers that report quality must surface this
        count alongside it, or a partially executed run reads as a genuine result.
    :ivar _chunk_size: sliding-window size in characters (private).
    :ivar _step: window stride, ``chunk_size - overlap`` (private).
    """

    def __init__(
        self,
        catalog: PIICategoryCatalog,
        client: LLMClient,
        *,
        confidence: float = DEFAULT_AI_CONFIDENCE,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> None:
        """Configure the detector; the id is derived from the client's model.

        :param catalog: closed catalog of allowed ``pii_type`` ids.
        :param client: the shared LLM client; its ``model`` names the detector.
        :param confidence: confidence stamped on every candidate, in ``[0, 1]``.
        :param chunk_size: sliding-window size in characters.
        :param overlap: overlap between consecutive windows, ``< chunk_size``.
        :raises ValueError: if ``overlap`` is not smaller than ``chunk_size``.
        """
        if overlap >= chunk_size:
            raise ValueError(f"overlap {overlap} must be smaller than chunk_size {chunk_size}")
        super().__init__(f"ai.{client.model}", DetectorKind.AI)
        self._client = client
        self._catalog = catalog
        self._parser = AIResponseParser(catalog)
        self._confidence = confidence
        self._chunk_size = chunk_size
        self._step = chunk_size - overlap
        self.chunks_seen = 0
        self.chunks_failed = 0

    def _prompt(self, chunk: str) -> str:
        """Build the user prompt embedding the closed catalog and one chunk."""
        catalog = "\n".join(f"- {c.id}: {c.label}" for c in self._catalog)
        return (
            f"Catalog of allowed pii_type ids:\n{catalog}\n\n"
            f"Document text (in Italian):\n{chunk}\n\n"
            "Find every occurrence of personal data (PII) in the text above. Report only "
            "values that appear verbatim in the text; do not infer, translate or invent. "
            "Assign only catalog ids that clearly correspond; if none clearly applies, do "
            "not report the value. Reply as a JSON array of objects "
            '{"value": <verbatim substring of the text>, "pii_type": <id>, '
            '"rationale": <short reason>}.'
        )

    def detect(self, text: str) -> list[PIICandidate]:
        """Analyse the whole text with the LLM and return located candidates.

        For every chunk the model is queried, its answer parsed, and each returned
        value located back in the chunk (all occurrences) to obtain reliable global
        spans; a value that does not occur verbatim is discarded (anti-hallucination).
        Candidates are de-duplicated by ``(start, end, pii_type)`` across the overlap
        regions. Consistent with recall-first (§2.5.2), nothing is discarded on
        confidence grounds.

        :param text: normalized document text to scan.
        :returns: located candidates, possibly empty; never ``None``. A model or
            runtime failure degrades to ``[]`` (or the chunks that did succeed).
        """
        candidates: list[PIICandidate] = []
        seen: set[tuple[int, int, str]] = set()
        warned = False
        for chunk_start in range(0, len(text), self._step):
            chunk = text[chunk_start : chunk_start + self._chunk_size]
            self.chunks_seen += 1
            try:
                parsed = self._parser.parse(
                    self._client.complete(self._prompt(chunk), system=_AI_SYSTEM_PROMPT)
                )
            except Exception:  # noqa: BLE001 — any LLM/parse failure is skipped, never raised
                self.chunks_failed += 1
                if not warned:
                    _logger.warning("AI detector chunk failed, skipping", exc_info=True)
                    warned = True
                parsed = []
            for value, pii_type, rationale in parsed:
                search_from = 0
                while (local := chunk.find(value, search_from)) != -1:
                    start = chunk_start + local
                    end = start + len(value)
                    search_from = local + 1
                    key = (start, end, pii_type)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        self.build_candidate(
                            text, TextSpan(start, end), pii_type, self._confidence,
                            rationale=rationale,
                        )
                    )
            if chunk_start + self._chunk_size >= len(text):
                break
        return candidates


def _bucket(document_id: str) -> int:
    """Map a document id to a stable, well-spread integer bucket.

    Uses a content hash (not :func:`hash`, which is salted per process) so the same
    document falls in the same bucket across runs and machines — the sampling is
    random-looking yet **reproducible**.

    :param document_id: the document identity to bucket.
    :returns: a non-negative integer derived from the id.
    """
    return int.from_bytes(hashlib.blake2b(document_id.encode("utf-8"), digest_size=8).digest())


@dataclass(frozen=True)
class AITriggerPolicy:
    """Which documents of a batch get the (costly) AI pass — a caller policy.

    The AI pass is seconds-to-minutes per document on CPU, so it is not run on the
    whole corpus at every scan. The rate is a single knob:

    - ``0`` — disabled (no AI);
    - ``1`` — every document;
    - ``N > 1`` — roughly one document in ``N``, chosen by a **stable hash** of the
      ``document_id`` (:func:`_bucket`), so the sampled subset is spread across the
      tree at random yet identical on a re-run (unlike an every-Nth index, it does
      not cluster by enumeration order, and unlike ``random`` it is reproducible).

    Stateless and deterministic: no cross-scan counter to lose on a restart.

    :ivar sampling_rate: the rate knob (``0`` off, ``1`` all, ``N`` one-in-``N``).
    """

    sampling_rate: int = 0

    @classmethod
    def from_env(cls) -> AITriggerPolicy:
        """Read the sampling rate from :data:`PII_AI_SAMPLING_RATE_ENV`.

        A missing, non-integer or negative value is read as ``0`` (disabled), so a
        misconfigured environment never turns the AI on by surprise.

        :returns: the policy configured from the environment.
        """
        raw = os.environ.get(PII_AI_SAMPLING_RATE_ENV, "")
        try:
            rate = int(raw)
        except ValueError:
            rate = 0
        return cls(sampling_rate=max(0, rate))

    @property
    def enabled(self) -> bool:
        """:returns: ``True`` if any AI runs (``sampling_rate > 0``)."""
        return self.sampling_rate > 0

    def selects(self, document_id: str) -> bool:
        """Tell whether a given document is sampled for the AI pass.

        :param document_id: the document's identity (its bucket is derived from it).
        :returns: ``True`` if the document gets the AI pass; ``False`` when disabled.
            ``rate == 1`` selects every document; ``rate == N`` selects ~1 in ``N``.
        """
        return self.enabled and _bucket(document_id) % self.sampling_rate == 0


def resolve_num_predict() -> int:
    """Resolve the per-chunk token cap for AI detection.

    Reads :data:`PII_LLM_NUM_PREDICT_ENV`, falling back to
    :data:`DEFAULT_AI_NUM_PREDICT`; a non-integer or non-positive value is
    ignored in favour of the default. Shared by :func:`build_ai_detector` and the
    multi-model benchmark so both measure the same bounded configuration.

    :returns: the token cap to pass as ``num_predict`` to the LLM client.
    """
    raw = os.environ.get(PII_LLM_NUM_PREDICT_ENV, "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_AI_NUM_PREDICT
    return value if value > 0 else DEFAULT_AI_NUM_PREDICT


def build_ai_detector(
    config_dir: Path | None = None, client: LLMClient | None = None
) -> LLMDetector:
    """Build an :class:`LLMDetector` from the packaged catalog and the environment.

    :param config_dir: directory holding ``categories.yaml``; defaults to the
        packaged :func:`~pii_detection.detection.config.default_config_dir`.
    :param client: the LLM client to drive; defaults to a fresh
        :class:`~pii_detection.llm.client.LLMClient` whose model comes from
        :data:`PII_LLM_MODEL_ENV` (then the client's own default chain), capped at
        :func:`resolve_num_predict` tokens per answer.
    :returns: the configured detector, with ``detector_id == "ai.<model>"``.
    :raises ConfigError: if ``categories.yaml`` is missing or malformed.
    """
    base = config_dir if config_dir is not None else default_config_dir()
    catalog = load_category_catalog(base / "categories.yaml")
    resolved = client if client is not None else LLMClient(
        model=os.environ.get(PII_LLM_MODEL_ENV), num_predict=resolve_num_predict()
    )
    return LLMDetector(catalog, resolved)


__all__ = [
    "DEFAULT_AI_CONFIDENCE",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "PII_LLM_MODEL_ENV",
    "PII_AI_SAMPLING_RATE_ENV",
    "PII_LLM_NUM_PREDICT_ENV",
    "DEFAULT_AI_NUM_PREDICT",
    "AIResponseParser",
    "LLMDetector",
    "AITriggerPolicy",
    "resolve_num_predict",
    "build_ai_detector",
]
