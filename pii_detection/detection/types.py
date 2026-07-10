"""Modello dati condiviso del livello di detection (blocco B4).

Fissato in ``doc/planning.md`` §"Modello dati comune". Il layer distingue tre
livelli di dato:

1. :class:`PIICandidate` — output grezzo di UN detector, pre-merge.
2. :class:`PIIMatch` — output del merge (le "PII unificate" che alimentano B5).
3. ``IstanzaPII`` / ``VariazionePII`` — persistenza (fuori scope, in B5/B6).

**Minimizzazione (§2.3.11).** I DTO detection-time possono portare il campo
``text`` (serve al merge e alla leggibilità del report) e vivono solo in memoria
per la durata dell'elaborazione del singolo documento. L'entità persistente non
contiene mai il valore della PII.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class DetectorKind(str, Enum):
    """Tecnica di rilevamento che ha prodotto un candidato.

    Vocabolario chiuso *architetturale* (tre tecniche), da non confondere con le
    categorie PII: ``pii_type`` è una stringa dichiarata in config e liberamente
    estensibile (§2.3.10), mentre l'insieme delle tecniche è fissato
    dall'architettura e aggiungerne una è attività da sviluppatore.

    :cvar REGEX: detector a espressioni regolari, config-driven.
    :cvar NER: detector NER zero-shot (GLiNER).
    :cvar AI: detector AI generativa selettiva (passata campionata).
    """

    REGEX = "regex"
    NER = "ner"
    AI = "ai"


class ConfirmationLevel(str, Enum):
    """Esito del merge per uno span. Vocabolario chiuso architetturale.

    :cvar SINGLE_SOURCE: rilevato da una sola fonte, senza overlap; mai scartato
        (recall-first, §2.5.2).
    :cvar DOUBLE_CONFIRMED: stesso ``pii_type`` confermato da regex e NER su span
        sovrapposti.
    :cvar CONFLICTING: span sovrapposti ma ``pii_type`` discordante; nessun
        arbitraggio automatico, la risoluzione è demandata a B5.
    :cvar AI_DISCOVERED: trovato dalla passata AI campionata, assente dalle altre
        fonti.
    """

    SINGLE_SOURCE = "single_source"
    DOUBLE_CONFIRMED = "double_confirmed"
    CONFLICTING = "conflicting"
    AI_DISCOVERED = "ai_discovered"


@dataclass(frozen=True)
class TextSpan:
    """Intervallo di caratteri nel testo normalizzato del documento.

    Immutabile. Gli offset sono half-open ``[start, end)`` sul testo
    normalizzato prodotto da B3.

    :ivar start: offset carattere iniziale, inclusivo (``>= 0``).
    :ivar end: offset carattere finale, esclusivo (``> start``).
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """Valida l'invariante dello span.

        :raises ValueError: se ``start`` è negativo, oppure se lo span è vuoto o
            invertito (``end <= start``).
        """
        if self.start < 0:
            raise ValueError(f"start negativo: {self.start}")
        if self.end <= self.start:
            raise ValueError(f"span vuoto o invertito: [{self.start}, {self.end})")

    def __len__(self) -> int:
        """:returns: numero di caratteri coperti dallo span."""
        return self.end - self.start

    def overlaps(self, other: TextSpan) -> bool:
        """Indica se i due span condividono almeno un carattere.

        Due span adiacenti (``self.end == other.start``) NON si sovrappongono.

        :param other: span con cui confrontarsi.
        :returns: ``True`` se l'intersezione è non vuota.
        """
        return self.start < other.end and other.start < self.end

    def overlap_ratio(self, other: TextSpan) -> float:
        """Intersection over Union (IoU) tra i due span.

        Metrica usata dal :class:`~pii_detection.detection.pipeline.MergeEngine`
        (Step 7) per decidere se due candidati si riferiscono allo stesso match.

        :param other: span con cui confrontarsi.
        :returns: rapporto in ``[0.0, 1.0]``; ``0.0`` se disgiunti, ``1.0`` se
            coincidono.
        """
        inter = max(0, min(self.end, other.end) - max(self.start, other.start))
        if inter == 0:
            return 0.0
        union = max(self.end, other.end) - min(self.start, other.start)
        return inter / union


@dataclass(frozen=True)
class DocumentLocation:
    """Posizione umana della PII, dalla mappa posizione fornita da B3.

    Tutti i campi sono opzionali perché dipendono dal formato sorgente: un PDF ha
    ``page``, un foglio di calcolo ha ``cell``, un testo semplice può non averne.

    :ivar page: numero di pagina (1-based), se applicabile.
    :ivar paragraph: indice di paragrafo, se applicabile.
    :ivar line: numero di riga, se applicabile.
    :ivar cell: riferimento di cella (es. ``"B4"``) per formati tabellari.
    """

    page: int | None = None
    paragraph: int | None = None
    line: int | None = None
    cell: str | None = None


@dataclass(frozen=True)
class NormalizedDocument:
    """Input del livello B4: testo normalizzato più identificativo del documento.

    La mappatura ``TextSpan -> DocumentLocation`` è responsabilità di B3 (fuori
    scope qui): finché non è disponibile, :meth:`location_for` restituisce
    ``None`` e la pipeline usa il valore ricevuto senza assumerne la presenza.

    :ivar document_id: identificativo stabile del documento sorgente.
    :ivar text: testo normalizzato su cui operano i detector.
    """

    document_id: str
    text: str

    def location_for(self, span: TextSpan) -> DocumentLocation | None:
        """Risolve la posizione umana di uno span.

        Placeholder in attesa di B3: la mappa di posizione la fornisce il layer
        di estrazione, non questo.

        :param span: intervallo di caratteri da localizzare.
        :returns: la :class:`DocumentLocation` corrispondente, o ``None`` finché
            B3 non fornisce la mappa.
        """
        return None


@dataclass(frozen=True)
class DetectionProvenance:
    """Provenienza di un singolo rilevamento — risponde alla tracciabilità (§2.7.3).

    Immutabile. I campi opzionali sono specifici della tecnica: ``raw_label`` per
    NER, ``checksum_validated`` per regex, ``rationale`` per AI. Mantenerli in un
    unico DTO permette a :class:`PIIMatch` di conservare provenienze eterogenee
    in un'unica lista ``sources``.

    :ivar detector_id: id dell'istanza di detector, es. ``"regex.iban_v1"``.
    :ivar detector_kind: tecnica che ha prodotto il rilevamento.
    :ivar pii_type: categoria PII dichiarata in config, es. ``"iban"``.
    :ivar confidence: confidenza del rilevamento in ``[0.0, 1.0]``.
    :ivar raw_label: solo NER — label testuale passata al modello.
    :ivar checksum_validated: solo regex — esito del validatore; ``None`` se
        nessun validatore è configurato per la regola.
    :ivar rationale: solo AI — motivazione testuale prodotta dal modello.
    """

    detector_id: str
    detector_kind: DetectorKind
    pii_type: str
    confidence: float
    raw_label: str | None = None
    checksum_validated: bool | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        """Valida il range di confidenza.

        :raises ValueError: se ``confidence`` è fuori da ``[0.0, 1.0]``.
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence fuori range [0,1]: {self.confidence}")


@dataclass
class PIICandidate:
    """Output di UN detector, pre-merge.

    Mutabile per comodità dei detector in fase di costruzione. Il campo ``text``
    vive solo in-memory (§2.3.11) e non deve mai raggiungere la persistenza.

    :ivar span: posizione del candidato nel testo normalizzato.
    :ivar text: sottostringa effettivamente rilevata (solo in-memory).
    :ivar provenance: da quale detector e con quale confidenza proviene.
    """

    span: TextSpan
    text: str
    provenance: DetectionProvenance


@dataclass
class PIIMatch:
    """PII unificata prodotta dal merge — output del layer B4 verso B5.

    Aggrega uno o più :class:`PIICandidate` che insistono sullo stesso span. La
    lista ``sources`` (non un singolo id) conserva tutte le provenienze: è ciò
    che alimenta :attr:`ConfirmationLevel.DOUBLE_CONFIRMED` e che il DPO può
    ispezionare (§2.7.3).

    :ivar span: posizione della PII nel testo normalizzato.
    :ivar text: valore rilevato (solo in-memory, §2.3.11).
    :ivar pii_type: categoria PII risultante dal merge.
    :ivar confidence: confidenza aggregata in ``[0.0, 1.0]``.
    :ivar confirmation_level: esito del merge per questo span.
    :ivar sources: provenienze che concorrono al match (almeno una).
    :ivar document_id: documento a cui la PII appartiene.
    :ivar location: posizione umana da B3, o ``None`` se non disponibile.
    :ivar match_id: identificativo detection-time (uuid4); non è una chiave
        persistente.
    """

    span: TextSpan
    text: str
    pii_type: str
    confidence: float
    confirmation_level: ConfirmationLevel
    sources: list[DetectionProvenance]
    document_id: str
    location: DocumentLocation | None = None
    match_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        """Valida confidenza e presenza di almeno una provenienza.

        :raises ValueError: se ``confidence`` è fuori da ``[0.0, 1.0]`` o se
            ``sources`` è vuota (violerebbe la tracciabilità, §2.7.3).
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence fuori range [0,1]: {self.confidence}")
        if not self.sources:
            raise ValueError("un PIIMatch deve avere almeno una provenienza (§2.7.3)")


__all__ = [
    "DetectorKind",
    "ConfirmationLevel",
    "TextSpan",
    "DocumentLocation",
    "NormalizedDocument",
    "DetectionProvenance",
    "PIICandidate",
    "PIIMatch",
]
