"""Document archetypes: bodies that grow in length without growing in fakeness.

A corpus is only a stress test if the long documents are *long the way real ones
are*. So a body is not one template stretched by repetition: it is composed of
**sections** — numbered prose sections drawn from a pool, interleaved with
PII-bearing blocks — until the target length for its
:class:`~pii_detection.evaluation.enterprise.types.SizeClass` is reached. A long
contract is a contract with more articles.

Two properties are deliberate:

* **PII is scattered through the body**, not stacked in the header. On a 40-page
  document that is what forces extraction and detection to work past page one.
* **Documents without PII carry distractors** — protocol numbers, amounts,
  product codes, meeting dates, company names. Prose that looks like it *could*
  hold PII is the only way a false positive can show up in the score at all.

Every PII value is emitted through
:func:`~pii_detection.evaluation.corpus_generator.annotate`, so the ground truth
is a by-product of writing the document, never a second thing to keep in sync.
The Italian prose is test *content*, not code: identifiers, archetype ids and
comments stay English.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from pii_detection.evaluation.corpus_generator import PIIValueFactory, annotate
from pii_detection.evaluation.enterprise.types import SizeClass

#: Builds one PII-bearing paragraph (a few lines) from the seeded value factory.
SectionBuilder = Callable[[PIIValueFactory], list[str]]

#: Target body length per size class, as ``(min_lines, max_lines)``. The PDF
#: renderer lays out roughly 30 lines per page, so the classes span a one-page
#: note to a 30-60 page manual.
SIZE_LINES: dict[SizeClass, tuple[int, int]] = {
    SizeClass.SHORT: (5, 15),
    SizeClass.MEDIUM: (40, 90),
    SizeClass.LONG: (120, 260),
    SizeClass.HUGE: (900, 1800),
}

#: Company names used as filler subjects. Legal persons are not personal data,
#: which is exactly why they belong in the distractor pool.
_COMPANIES = (
    "Delta Servizi S.r.l.", "Nordest Logistica S.p.A.", "Aurora Consulting",
    "Blu Ricambi S.r.l.", "Tecnoimpianti Adriatica", "Gamma Forniture",
    "Sistemi Integrati Cusio", "Verdi Manutenzioni S.n.c.",
)

#: Placeholder values a filler template may ask for. All plausible-looking
#: numbers that are **not** PII: the bait for false positives.
_DISTRACTORS: dict[str, Callable[[random.Random], str]] = {
    "protocol": lambda r: f"{r.randint(1000, 9999)}/{r.randint(2019, 2025)}",
    "amount": lambda r: f"{r.randint(1, 90)}.{r.randint(0, 999):03d},{r.randint(0, 99):02d}",
    "code": lambda r: f"ART-{r.randint(100, 999)}-{r.choice('ABCDEFGH')}",
    "order": lambda r: f"ODA-{r.randint(2019, 2025)}-{r.randint(1000, 9999)}",
    "meeting_date": lambda r: f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/{r.randint(2019, 2025)}",
    "percent": lambda r: f"{r.randint(2, 40)}%",
    "company": lambda r: r.choice(_COMPANIES),
    "version": lambda r: f"{r.randint(1, 4)}.{r.randint(0, 9)}",
    "room": lambda r: f"sala {r.choice(('A', 'B', 'C'))}{r.randint(1, 4)}",
    "hour": lambda r: f"{r.randint(8, 18):02d}:{r.choice(('00', '15', '30', '45'))}",
    "days": lambda r: str(r.randint(5, 90)),
    "quantity": lambda r: str(r.randint(10, 5000)),
}


def _fill(template: str, rng: random.Random) -> list[str]:
    """Resolve a filler template's distractor placeholders into lines.

    :param template: prose with ``{placeholder}`` fields from :data:`_DISTRACTORS`.
    :param rng: seeded RNG, so the same plan always yields the same prose.
    :returns: the resolved text, split into lines.
    """
    values = {name: build(rng) for name, build in _DISTRACTORS.items()}
    return template.format(**values).split("\n")


# --- PII-bearing blocks -------------------------------------------------------
# Small, reusable paragraphs; archetypes compose them. Keeping them separate is
# what lets a "long contract" and a "short record" share the same identity block
# without duplicating the markers.


def _b_employee_identity(f: PIIValueFactory) -> list[str]:
    """Identity of an employee: person_name, date_of_birth, italian_id."""
    return [
        f"Dipendente: {annotate('person_name', f.person_name())}",
        f"Nato/a il: {annotate('date_of_birth', f.date_of_birth())}",
        f"Codice fiscale: {annotate('italian_id', f.italian_id())}",
    ]


def _b_residence(f: PIIValueFactory) -> list[str]:
    """Declared residence: address."""
    return [f"Residenza dichiarata: {annotate('address', f.address())}"]


def _b_contacts(f: PIIValueFactory) -> list[str]:
    """Contact details: phone, email."""
    return [
        f"Recapito telefonico: {annotate('phone', f.phone())}",
        f"Indirizzo di posta elettronica: {annotate('email', f.email())}",
    ]


def _b_salary(f: PIIValueFactory) -> list[str]:
    """Payroll coordinates: iban."""
    return [
        "La retribuzione e' accreditata sul conto corrente indicato dal dipendente:",
        f"IBAN {annotate('iban', f.iban())}.",
    ]


def _b_cross_border(f: PIIValueFactory) -> list[str]:
    """Swiss cross-border worker: person_name, swiss_avs."""
    return [
        f"Lavoratore frontaliero: {annotate('person_name', f.person_name())}",
        f"Numero AVS: {annotate('swiss_avs', f.swiss_avs())}",
    ]


def _b_occupational_health(f: PIIValueFactory) -> list[str]:
    """Occupational-medicine note: person_name, health_data (special category)."""
    return [
        f"Visita periodica di {annotate('person_name', f.person_name())}.",
        f"Quadro clinico rilevato: {annotate('health_data', f.health_data())}.",
        "Giudizio di idoneita' con prescrizioni, da rivalutare al prossimo controllo.",
    ]


def _b_card_payment(f: PIIValueFactory) -> list[str]:
    """Card payment attempt: credit_card."""
    return [
        f"Pagamento tentato con la carta {annotate('credit_card', f.credit_card())},",
        "transazione respinta dal circuito.",
    ]


def _b_client_reference(f: PIIValueFactory) -> list[str]:
    """Client on a commercial document: person_name, address."""
    return [
        f"Intestatario: {annotate('person_name', f.person_name())}",
        f"Indirizzo di fatturazione: {annotate('address', f.address())}",
    ]


def _b_billing_id(f: PIIValueFactory) -> list[str]:
    """Billing tax id: italian_id."""
    return [f"Codice fiscale dell'intestatario: {annotate('italian_id', f.italian_id())}"]


def _b_iban_payment(f: PIIValueFactory) -> list[str]:
    """Payment instruction: iban."""
    return [f"Il pagamento va disposto sull'IBAN {annotate('iban', f.iban())}."]


def _b_access_entry(f: PIIValueFactory) -> list[str]:
    """Access-log entry: ip_address, email."""
    return [
        f"Accesso registrato dall'indirizzo IP {annotate('ip_address', f.ip_address())} "
        f"con l'utenza {annotate('email', f.email())}."
    ]


def _b_incident_contact(f: PIIValueFactory) -> list[str]:
    """Incident reporter: person_name, email, ip_address."""
    return [
        f"Segnalazione a cura di {annotate('person_name', f.person_name())} "
        f"({annotate('email', f.email())}).",
        f"Origine del traffico anomalo: {annotate('ip_address', f.ip_address())}.",
    ]


def _b_customer_contact(f: PIIValueFactory) -> list[str]:
    """Customer contact details: email, phone."""
    return [
        f"Contatto del cliente: {annotate('email', f.email())} — "
        f"{annotate('phone', f.phone())}."
    ]


def _b_registry_row(f: PIIValueFactory) -> list[str]:
    """One export row: person_name, email, phone, italian_id."""
    return [
        f"{annotate('person_name', f.person_name())}; {annotate('email', f.email())}; "
        f"{annotate('phone', f.phone())}; {annotate('italian_id', f.italian_id())}"
    ]


def _b_newsletter_row(f: PIIValueFactory) -> list[str]:
    """One marketing-list row: person_name, email, date_of_birth."""
    return [
        f"{annotate('person_name', f.person_name())}; {annotate('email', f.email())}; "
        f"nato/a il {annotate('date_of_birth', f.date_of_birth())}"
    ]


def _b_candidate(f: PIIValueFactory) -> list[str]:
    """Job applicant: person_name, date_of_birth, address, email, phone."""
    return [
        f"Candidato/a: {annotate('person_name', f.person_name())}",
        f"Data di nascita: {annotate('date_of_birth', f.date_of_birth())}",
        f"Domicilio: {annotate('address', f.address())}",
        f"Contatti: {annotate('email', f.email())}, {annotate('phone', f.phone())}",
    ]


# --- Filler pools -------------------------------------------------------------

_FILLER_HR = (
    "Le parti danno atto che l'inquadramento contrattuale segue il CCNL di categoria,\n"
    "con decorrenza dalla data di sottoscrizione e periodo di prova di {days} giorni.\n"
    "Eventuali variazioni sono comunicate per iscritto al protocollo {protocol}.",
    "L'orario di lavoro e' articolato su cinque giornate settimanali, con ingresso\n"
    "flessibile entro le {hour}. Le ore eccedenti sono recuperate entro il trimestre\n"
    "successivo, salvo diversa intesa con il responsabile di funzione.",
    "La formazione obbligatoria in materia di sicurezza e' erogata entro {days} giorni\n"
    "dall'inserimento. Il registro delle presenze ai corsi e' conservato dall'ufficio\n"
    "del personale insieme agli attestati rilasciati.",
    "Le ferie maturano in ragione di un dodicesimo per ogni mese di servizio prestato.\n"
    "La pianificazione e' concordata con il responsabile entro il mese di aprile,\n"
    "compatibilmente con le esigenze organizzative del reparto.",
)

_FILLER_LEGAL = (
    "Le parti si impegnano a mantenere riservata ogni informazione acquisita\n"
    "nell'esecuzione del rapporto, anche successivamente alla sua cessazione,\n"
    "per un periodo non inferiore a {days} mesi.",
    "Il presente atto e' regolato dalla legge italiana. Per ogni controversia\n"
    "e' competente in via esclusiva il foro della sede legale del committente,\n"
    "con espressa rinuncia a fori alternativi.",
    "L'inadempimento di una delle obbligazioni essenziali legittima la risoluzione\n"
    "di diritto ai sensi dell'art. 1456 c.c., previa diffida ad adempiere nel\n"
    "termine di {days} giorni.",
    "Il corrispettivo pattuito, pari a euro {amount}, e' comprensivo di ogni onere\n"
    "accessorio; eventuali revisioni sono ammesse nel limite del {percent} annuo.",
    "Ogni modifica al presente accordo richiede la forma scritta a pena di nullita'.\n"
    "Le comunicazioni si intendono validamente effettuate se inviate agli indirizzi\n"
    "indicati in epigrafe e riscontrate entro {days} giorni.",
)

_FILLER_TECH = (
    "Il componente e' distribuito nella versione {version} e richiede la libreria\n"
    "di runtime installata sui nodi applicativi. L'aggiornamento e' incrementale e\n"
    "non richiede fermo del servizio.",
    "In caso di errore il sistema scrive una voce nel log applicativo con il codice\n"
    "{code}. Le voci sono ruotate ogni {days} giorni e archiviate in sola lettura.",
    "La procedura di ripristino prevede il recupero dell'ultimo backup consistente,\n"
    "la verifica dell'integrita' degli indici e il riavvio ordinato dei servizi\n"
    "dipendenti secondo la sequenza documentata.",
    "Le soglie di allarme sono configurate al {percent} di occupazione delle risorse.\n"
    "Il superamento genera una notifica al gruppo di reperibilita' e apre un ticket\n"
    "automatico sulla coda di secondo livello.",
)

_FILLER_MINUTES = (
    "Alle ore {hour} in {room} si apre la seduta. Il presidente illustra lo stato\n"
    "di avanzamento delle attivita' deliberate nella riunione del {meeting_date}.",
    "Si discute la proposta di rinnovo del contratto con {company}, per un valore\n"
    "stimato di euro {amount}. Il consiglio rinvia la decisione alla prossima seduta,\n"
    "chiedendo un confronto con almeno due offerte alternative.",
    "Viene approvato all'unanimita' il piano di manutenzione, con uno scostamento\n"
    "massimo del {percent} rispetto al preventivo protocollato al n. {protocol}.",
    "Nessun altro argomento essendo posto in discussione, la seduta e' tolta.\n"
    "Il verbale e' letto e approvato seduta stante.",
)

_FILLER_COMMERCIAL = (
    "L'offerta ha validita' di {days} giorni dalla data di emissione e si intende\n"
    "accettata con la trasmissione dell'ordine di acquisto {order}.",
    "I prezzi indicati sono al netto di IVA e si riferiscono a forniture rese franco\n"
    "magazzino. Per quantitativi superiori a {quantity} pezzi si applica uno sconto\n"
    "del {percent}.",
    "La consegna e' prevista entro {days} giorni lavorativi dalla conferma d'ordine,\n"
    "salvo indisponibilita' temporanea dei materiali comunicata tempestivamente.",
    "Articolo {code} — fornitura standard, imballo compreso, prezzo unitario\n"
    "euro {amount} per lotti minimi di {quantity} unita'.",
)

_FILLER_IT = (
    "La rotazione delle credenziali di servizio e' pianificata ogni {days} giorni.\n"
    "Le utenze applicative non interattive sono escluse dalla scadenza automatica\n"
    "ma soggette a revisione periodica.",
    "Il perimetro monitorato comprende i sistemi esposti e i servizi interni critici.\n"
    "Gli eventi sono correlati per finestra temporale e classificati per severita'.",
    "L'incidente e' stato classificato di severita' media e chiuso entro i tempi\n"
    "previsti dal contratto di servizio, senza impatto sui dati trattati.",
)

_FILLER_CORPORATE = (
    "Il presente documento e' emesso a cura della direzione competente e sostituisce\n"
    "ogni precedente versione. La revisione corrente e' la {version}.",
    "Le funzioni aziendali coinvolte adottano le misure organizzative necessarie e ne\n"
    "verificano l'efficacia con cadenza almeno annuale.",
    "Eventuali deroghe sono autorizzate per iscritto e tracciate con il numero di\n"
    "protocollo {protocol}, con indicazione della motivazione e della durata.",
)


@dataclass(frozen=True)
class Archetype:
    """A kind of document: how it opens, what PII it carries, how it fills pages.

    :ivar kind: stable id used in the manifest and by the folder profiles.
    :ivar title: opening line, a filler template (distractors allowed).
    :ivar blocks: the PII-bearing paragraphs this kind may contain; empty for
        the PII-free archetypes, which exist to measure false positives.
    :ivar fillers: prose pool the body is padded with, in section form.
    """

    kind: str
    title: str
    blocks: tuple[SectionBuilder, ...]
    fillers: tuple[str, ...]

    @property
    def has_pii(self) -> bool:
        """:returns: whether documents of this kind contain any PII at all."""
        return bool(self.blocks)


#: Every archetype, by id. The PII-free ones (``policy``, ``tech_manual``,
#: ``price_list``, ``meeting_minutes``, ``campaign_brief``,
#: ``supplier_contract``) are as load-bearing as the others: without them the
#: corpus cannot say anything about precision.
ARCHETYPES: dict[str, Archetype] = {
    "hr_contract": Archetype(
        "hr_contract",
        "CONTRATTO INDIVIDUALE DI LAVORO — prot. {protocol}",
        (_b_employee_identity, _b_residence, _b_contacts, _b_salary),
        _FILLER_HR + _FILLER_LEGAL,
    ),
    "hr_record": Archetype(
        "hr_record",
        "Scheda di assunzione — pratica {protocol}",
        (_b_employee_identity, _b_residence, _b_contacts),
        _FILLER_HR,
    ),
    "payslip": Archetype(
        "payslip",
        "Cedolino paga — periodo di competenza {meeting_date}",
        (_b_employee_identity, _b_salary),
        _FILLER_HR,
    ),
    "cv": Archetype(
        "cv",
        "Curriculum vitae — candidatura {protocol}",
        (_b_candidate,),
        _FILLER_HR + _FILLER_CORPORATE,
    ),
    "interview_note": Archetype(
        "interview_note",
        "Nota di colloquio — {meeting_date}, {room}",
        (_b_candidate,),
        _FILLER_HR,
    ),
    "occupational_health": Archetype(
        "occupational_health",
        "Sorveglianza sanitaria — verbale {protocol}",
        (_b_occupational_health, _b_employee_identity),
        _FILLER_HR + _FILLER_CORPORATE,
    ),
    "cross_border_notice": Archetype(
        "cross_border_notice",
        "Notifica lavoratori frontalieri — prot. {protocol}",
        (_b_cross_border, _b_salary, _b_residence),
        _FILLER_HR + _FILLER_LEGAL,
    ),
    "invoice": Archetype(
        "invoice",
        "Fattura n. {protocol} del {meeting_date}",
        (_b_client_reference, _b_billing_id, _b_iban_payment),
        _FILLER_COMMERCIAL,
    ),
    "payment_notice": Archetype(
        "payment_notice",
        "Avviso di pagamento — pratica {protocol}",
        (_b_iban_payment, _b_customer_contact),
        _FILLER_COMMERCIAL,
    ),
    "supplier_invoice": Archetype(
        "supplier_invoice",
        "Fattura fornitore {company} — ordine {order}",
        (_b_iban_payment,),
        _FILLER_COMMERCIAL,
    ),
    "client_letter": Archetype(
        "client_letter",
        "Comunicazione al cliente — rif. {protocol}",
        (_b_client_reference, _b_customer_contact),
        _FILLER_COMMERCIAL + _FILLER_CORPORATE,
    ),
    "quotation": Archetype(
        "quotation",
        "Offerta commerciale {order}",
        (_b_customer_contact,),
        _FILLER_COMMERCIAL,
    ),
    "payment_dispute": Archetype(
        "payment_dispute",
        "Reclamo pagamento — ticket {protocol}",
        (_b_card_payment, _b_customer_contact),
        _FILLER_COMMERCIAL + _FILLER_CORPORATE,
    ),
    "client_registry_export": Archetype(
        "client_registry_export",
        "Estratto anagrafica clienti (export del {meeting_date})",
        (_b_registry_row, _b_registry_row, _b_registry_row),
        _FILLER_CORPORATE,
    ),
    "newsletter_export": Archetype(
        "newsletter_export",
        "Lista iscritti newsletter — segmento {code}",
        (_b_newsletter_row, _b_newsletter_row, _b_newsletter_row),
        _FILLER_COMMERCIAL,
    ),
    "access_log": Archetype(
        "access_log",
        "Registro accessi applicativi — estrazione {meeting_date}",
        (_b_access_entry, _b_access_entry),
        _FILLER_IT,
    ),
    "incident_report": Archetype(
        "incident_report",
        "Rapporto di incidente {protocol}",
        (_b_incident_contact,),
        _FILLER_IT + _FILLER_TECH,
    ),
    # --- PII-free archetypes: the precision half of the corpus ---
    "policy": Archetype(
        "policy",
        "Procedura interna — revisione {version}",
        (),
        _FILLER_CORPORATE + _FILLER_LEGAL,
    ),
    "tech_manual": Archetype(
        "tech_manual",
        "Manuale operativo del sistema — versione {version}",
        (),
        _FILLER_TECH + _FILLER_CORPORATE,
    ),
    "price_list": Archetype(
        "price_list",
        "Listino prezzi — edizione {version}",
        (),
        _FILLER_COMMERCIAL,
    ),
    "meeting_minutes": Archetype(
        "meeting_minutes",
        "Verbale della riunione del {meeting_date}",
        (),
        _FILLER_MINUTES + _FILLER_CORPORATE,
    ),
    "campaign_brief": Archetype(
        "campaign_brief",
        "Brief di campagna — codice {code}",
        (),
        _FILLER_COMMERCIAL + _FILLER_CORPORATE,
    ),
    "supplier_contract": Archetype(
        "supplier_contract",
        "Contratto di fornitura con {company} — prot. {protocol}",
        (),
        _FILLER_LEGAL + _FILLER_COMMERCIAL,
    ),
}

#: Ids of the archetypes that contain no PII at all.
PII_FREE_KINDS: tuple[str, ...] = tuple(
    kind for kind, archetype in ARCHETYPES.items() if not archetype.has_pii
)

#: Probability of inserting one more PII block once every declared block has
#: been used. Keeps PII appearing deep into long documents instead of only in
#: the opening page.
_EXTRA_BLOCK_PROBABILITY = 0.35

#: Section headings cycled in long bodies, so a 40-page document does not repeat
#: the same title 200 times.
_SECTION_HEADINGS = (
    "Oggetto e ambito di applicazione",
    "Modalita' operative",
    "Responsabilita'",
    "Termini e condizioni",
    "Disposizioni generali",
    "Verifiche e controlli",
    "Documentazione di riferimento",
    "Note conclusive",
)


def target_lines(size_class: SizeClass, rng: random.Random) -> int:
    """Draw a body length for a size class.

    :param size_class: the length band.
    :param rng: seeded RNG.
    :returns: the number of lines to aim for.
    """
    low, high = SIZE_LINES[size_class]
    return rng.randint(low, high)


def build_body(archetype: Archetype, factory: PIIValueFactory, lines_target: int) -> str:
    """Compose one annotated document body of roughly the requested length.

    Every declared PII block is emitted at least once — a short document that
    dropped its blocks would be a document whose gold silently shrank — after
    which sections and further blocks alternate until the target is reached, so
    PII keeps appearing to the last page.

    :param archetype: the kind of document to write.
    :param factory: seeded value factory (its ``rng`` also drives the layout).
    :param lines_target: desired number of lines.
    :returns: the body with ``{{pii_type:value}}`` markers still in place.
    """
    rng = factory.rng
    lines: list[str] = _fill(archetype.title, rng)
    lines.append("")
    pending = list(archetype.blocks)
    section = 0
    while len(lines) < lines_target or pending:
        take_block = bool(pending) and (
            not archetype.fillers or rng.random() < 0.5 or len(lines) + 6 >= lines_target
        )
        if take_block:
            lines.extend(pending.pop(0)(factory))
        elif archetype.fillers:
            section += 1
            lines.append(f"{section}. {rng.choice(_SECTION_HEADINGS)}")
            lines.extend(_fill(rng.choice(archetype.fillers), rng))
            if archetype.blocks and rng.random() < _EXTRA_BLOCK_PROBABILITY:
                lines.extend(rng.choice(archetype.blocks)(factory))
        else:  # pragma: no cover - every archetype declares fillers or blocks
            break
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ARCHETYPES",
    "PII_FREE_KINDS",
    "SIZE_LINES",
    "Archetype",
    "SectionBuilder",
    "build_body",
    "target_lines",
]
