"""Presidio-backed detectors behind the ``PIIDetector`` contract.

Wraps Microsoft Presidio as a detector without letting it drive the pipeline
(design ``doc/sections/6_implementazione.tex`` §"valutazione-presidio"): its
``AnalyzerEngine`` becomes the engine, this project's data model and merge stay
on top.

Presidio mixes pattern/checksum recognizers and NER in one engine. To keep the
two techniques as **independent sources** — which the merge's ``DOUBLE_CONFIRMED``
depends on — :func:`build_presidio_detectors` builds two
:class:`PresidioDetector` instances from the *same* analyzer: one tagged
:attr:`~pii_detection.detection.types.DetectorKind.REGEX` keeping the pattern
results, one tagged ``NER`` keeping the NER results. The split is by the name of
the recognizer that produced each result (:data:`NER_RECOGNIZER_NAMES`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

from pii_detection.detection.config import PresidioEntityModel
from pii_detection.detection.protocol import BaseDetector
from pii_detection.detection.types import DetectorKind, PIICandidate, TextSpan

#: Names of the Presidio recognizers whose results count as NER (vs pattern).
NER_RECOGNIZER_NAMES: frozenset[str] = frozenset(
    {"SpacyRecognizer", "StanzaRecognizer", "TransformersRecognizer", "GLiNERRecognizer"}
)

_RECOGNIZER_NAME_KEY = "recognizer_name"

#: Default GLiNER (zero-shot) label -> Presidio ``entity_type`` mapping. The keys
#: are the labels prompted to GLiNER; the values are Presidio entities, routed to
#: the ``pii_type`` catalog by ``presidio_entities.yaml`` (no duplicated mapping).
_GLINER_ENTITY_MAPPING: dict[str, str] = {
    "person": "PERSON",
    "full name": "PERSON",
    "address": "LOCATION",
    "email": "EMAIL_ADDRESS",
    "phone number": "PHONE_NUMBER",
    "credit card number": "CREDIT_CARD",
    "iban": "IBAN_CODE",
    "ip address": "IP_ADDRESS",
    "date of birth": "DATE_TIME",
}


def build_italian_analyzer(
    *,
    model_name: str = "it_core_news_lg",
    language: str = "it",
    swiss_avs_pattern: str | None = None,
    use_gliner: bool = False,
    gliner_model: str = "urchade/gliner_multi_pii-v1",
    gliner_threshold: float = 0.3,
) -> AnalyzerEngine:
    """Build an ``AnalyzerEngine`` wired to an installed Italian spaCy model.

    A bare ``AnalyzerEngine()`` defaults to English and tries to download
    ``en_core_web_lg`` at runtime; this factory configures the NLP engine
    explicitly on an already-installed Italian model instead.

    :param model_name: installed spaCy model to load, e.g. ``"it_core_news_lg"``.
    :param language: language code the analyzer answers on.
    :param swiss_avs_pattern: optional regex for the Swiss AVS number; when given,
        a custom ``PatternRecognizer`` for entity ``"SWISS_AVS"`` is registered
        (Presidio has no built-in for it).
    :param use_gliner: when ``True``, replace the spaCy NER recognizer with a
        GLiNER one (PII-specific zero-shot): spaCy stays only for tokenization,
        the NER comes from GLiNER alone. Needs the heavy ``[ner]`` dependencies;
        ``GLiNERRecognizer`` is imported lazily so the rest works without them.
    :param gliner_model: HuggingFace id of the GLiNER model to load.
    :param gliner_threshold: minimum GLiNER score to keep a span (recall-first: low).
    :returns: a configured analyzer ready for :meth:`AnalyzerEngine.analyze`.
    """
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": model_name}],
        }
    )
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(), supported_languages=[language]
    )
    if swiss_avs_pattern is not None:
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="SWISS_AVS",
                patterns=[Pattern("swiss_avs", swiss_avs_pattern, 0.6)],
                supported_language=language,
            )
        )
    if use_gliner:
        # Lazy import: GLiNER pulls torch, absent from the light local venv.
        from presidio_analyzer.predefined_recognizers import GLiNERRecognizer

        # NER from GLiNER only: drop Presidio's spaCy NER recognizer.
        analyzer.registry.remove_recognizer("SpacyRecognizer")
        analyzer.registry.add_recognizer(
            GLiNERRecognizer(
                supported_language=language,
                entity_mapping=_GLINER_ENTITY_MAPPING,
                model_name=gliner_model,
                threshold=gliner_threshold,
                map_location="cpu",
            )
        )
    return analyzer


class PresidioDetector(BaseDetector):
    """One detector over a shared Presidio ``AnalyzerEngine``.

    The :attr:`detector_kind` both labels the provenance and selects which
    results are kept: ``REGEX`` keeps the pattern/checksum recognizers, ``NER``
    keeps the NER recognizers (per :data:`NER_RECOGNIZER_NAMES`).

    :ivar detector_id: identifier of the instance, e.g. ``"presidio.pattern"``.
    :ivar detector_kind: technique this instance represents.
    """

    def __init__(
        self,
        detector_id: str,
        detector_kind: DetectorKind,
        analyzer: AnalyzerEngine,
        entity_map: Mapping[str, str],
        *,
        language: str = "it",
        ner_recognizer_names: frozenset[str] = NER_RECOGNIZER_NAMES,
    ) -> None:
        """Store the identity, the shared analyzer and the entity mapping.

        :param detector_id: unique identifier of the instance.
        :param detector_kind: ``REGEX`` to keep pattern results, ``NER`` to keep
            NER results.
        :param analyzer: shared, already-configured Presidio analyzer.
        :param entity_map: Presidio ``entity_type``→``pii_type`` mapping; entities
            absent from it are dropped.
        :param language: language passed to :meth:`AnalyzerEngine.analyze`.
        :param ner_recognizer_names: recognizer names whose results count as NER.
        """
        super().__init__(detector_id, detector_kind)
        self._analyzer = analyzer
        self._entity_map = dict(entity_map)
        self._language = language
        self._ner_names = ner_recognizer_names
        self._keep_ner = detector_kind is DetectorKind.NER

    def detect(self, text: str) -> list[PIICandidate]:
        """Run Presidio and map the results of this technique into candidates.

        Consistent with recall-first (§2.5.2), every mapped result is kept at the
        score Presidio assigned. Results of the other technique, unmapped entity
        types, and zero-width spans are skipped.

        :param text: normalized document text to scan.
        :returns: one candidate per kept result, possibly empty; never ``None``.
        """
        candidates: list[PIICandidate] = []
        for result in self._analyzer.analyze(text=text, language=self._language):
            metadata = result.recognition_metadata or {}
            is_ner = metadata.get(_RECOGNIZER_NAME_KEY) in self._ner_names
            if is_ner != self._keep_ner:
                continue
            pii_type = self._entity_map.get(result.entity_type)
            if pii_type is None or result.start == result.end:
                continue
            raw_label = result.entity_type if self._keep_ner else None
            candidates.append(
                self.build_candidate(
                    text,
                    TextSpan(result.start, result.end),
                    pii_type,
                    result.score,
                    raw_label=raw_label,
                )
            )
        return candidates


def build_presidio_detectors(
    entities: Iterable[PresidioEntityModel],
    analyzer: AnalyzerEngine,
    *,
    language: str = "it",
) -> tuple[PresidioDetector, PresidioDetector]:
    """Build the pattern and NER detectors from a shared analyzer and mapping.

    :param entities: validated ``entity_type``→``pii_type`` mappings loaded from
        ``presidio_entities.yaml``.
    :param analyzer: shared, already-configured Presidio analyzer.
    :param language: language passed to the detectors.
    :returns: a ``(pattern_detector, ner_detector)`` pair sharing the analyzer.
    """
    entity_map = {e.entity: e.pii_type for e in entities}
    pattern = PresidioDetector(
        "presidio.pattern", DetectorKind.REGEX, analyzer, entity_map, language=language
    )
    ner = PresidioDetector(
        "presidio.ner", DetectorKind.NER, analyzer, entity_map, language=language
    )
    return pattern, ner


__all__ = [
    "NER_RECOGNIZER_NAMES",
    "build_italian_analyzer",
    "PresidioDetector",
    "build_presidio_detectors",
]
