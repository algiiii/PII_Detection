"""Synthetic evaluation-corpus generator for the detection layer (block B4).

Produces inline-annotated documents in the ``{{pii_type:value}}`` format read by
:mod:`pii_detection.evaluation.corpus`, so the ground truth comes for free: the
loader derives the gold spans from the markers, nobody counts offsets by hand.

The PII values are realistic and **checksum-valid** wherever the category has a
checksum (IBAN, credit card, Swiss AVS, Italian tax code). This matters because
Presidio's recognizers validate those checksums: an invalid value would be
silently rejected and depress recall, turning the generator itself into the
reason a detector "misses". :func:`validate_factory` re-checks the values with
pure algorithms, independently of Presidio.

The same generated content later feeds the end-to-end (Tier 2) corpus once
rendered to PDF/DOCX; this module only emits the annotated text.

Requires the ``[eval]`` optional dependencies (``faker``,
``python-codicefiscale``), both pure-Python: the corpus builds in the local venv
without the heavy ML stack.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Literal

from codicefiscale import codicefiscale
from faker import Faker

from pii_detection.evaluation.corpus import parse_annotated_text

#: Italian cities the codice-fiscale encoder recognises (Belfiore registry).
_CF_CITIES = (
    "Roma", "Milano", "Napoli", "Torino", "Genova",
    "Bologna", "Firenze", "Palermo", "Bari", "Venezia",
)

#: Free-text health conditions (special category, GDPR art. 9) for ``health_data``.
_HEALTH_CONDITIONS = (
    "diabete di tipo 2", "ipertensione arteriosa", "asma bronchiale",
    "epatite C", "cardiopatia ischemica", "insufficienza renale cronica",
    "sindrome ansioso-depressiva", "ipotiroidismo", "positività HIV",
    "morbo di Crohn",
)


def _iban_is_valid(iban: str) -> bool:
    """Validate an IBAN with the ISO 7064 mod-97 rule.

    :param iban: IBAN string (spaces tolerated).
    :returns: ``True`` if the check digits are correct.
    """
    stripped = iban.replace(" ", "").upper()
    rearranged = stripped[4:] + stripped[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(digits) % 97 == 1


def _luhn_is_valid(number: str) -> bool:
    """Validate a number string with the Luhn algorithm (credit cards).

    :param number: digit string (non-digits ignored).
    :returns: ``True`` if the Luhn checksum holds.
    """
    digits = [int(d) for d in number if d.isdigit()]
    checksum = 0
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _ean13_check_digit(twelve: str) -> str:
    """Return the EAN-13 check digit of 12 digits (used by the Swiss AVS).

    :param twelve: exactly 12 digit characters.
    :returns: the single check-digit character.
    """
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(twelve))
    return str((10 - total % 10) % 10)


def _swiss_avs(rng: random.Random) -> str:
    """Build a valid Swiss AVS number ``756.XXXX.XXXX.XX`` with EAN-13 check."""
    body = "756" + "".join(str(rng.randint(0, 9)) for _ in range(9))  # 12 digits
    full = body + _ean13_check_digit(body)  # 13 digits
    return f"{full[0:3]}.{full[3:7]}.{full[7:11]}.{full[11:13]}"


class PIIValueFactory:
    """Produces realistic, checksum-valid values, one method per ``pii_type``.

    Seeded for reproducibility. Each method returns a plain value string, with no
    annotation markers (the templates wrap it).

    :ivar fake: the seeded Faker instance (``it_IT`` locale).
    :ivar rng: the seeded RNG for values Faker does not cover (AVS, choices).
    """

    def __init__(self, seed: int) -> None:
        """Seed both the Faker instance and the auxiliary RNG.

        :param seed: seed making the whole generation reproducible.
        """
        self.fake = Faker("it_IT")
        self.fake.seed_instance(seed)
        self.rng = random.Random(seed)

    def email(self) -> str:
        """:returns: a plausible e-mail address."""
        return str(self.fake.ascii_email())

    def phone(self) -> str:
        """:returns: an Italian-formatted phone number."""
        return str(self.fake.phone_number())

    def iban(self) -> str:
        """:returns: a mod-97-valid IBAN."""
        return str(self.fake.iban())

    def credit_card(self) -> str:
        """:returns: a Luhn-valid credit-card number."""
        return str(self.fake.credit_card_number())

    def ip_address(self) -> str:
        """:returns: an IPv4 address."""
        return str(self.fake.ipv4())

    def person_name(self) -> str:
        """:returns: a first + last name (no title prefix)."""
        return f"{self.fake.first_name()} {self.fake.last_name()}"

    def address(self) -> str:
        """:returns: a single-line Italian street address with postcode and city."""
        return f"{self.fake.street_address()}, {self.fake.postcode()} {self.fake.city()}"

    def date_of_birth(self) -> str:
        """:returns: a date of birth as ``dd/mm/yyyy``."""
        born: date = self.fake.date_of_birth(minimum_age=18, maximum_age=90)
        return born.strftime("%d/%m/%Y")

    def swiss_avs(self) -> str:
        """:returns: a valid Swiss AVS number."""
        return _swiss_avs(self.rng)

    def health_data(self) -> str:
        """:returns: a free-text health condition (special category)."""
        return self.rng.choice(_HEALTH_CONDITIONS)

    def italian_id(self) -> str:
        """:returns: a valid ``codice fiscale`` encoded from a fake identity."""
        gender: Literal["M", "F"] = "M" if self.rng.random() < 0.5 else "F"
        first = (
            self.fake.first_name_male() if gender == "M" else self.fake.first_name_female()
        )
        born: date = self.fake.date_of_birth(minimum_age=18, maximum_age=90)
        return str(
            codicefiscale.encode(
                lastname=self.fake.last_name(),
                firstname=first,
                gender=gender,
                birthdate=born.isoformat(),
                birthplace=self.rng.choice(_CF_CITIES),
            )
        )


def _ann(pii_type: str, value: str) -> str:
    """Wrap a value in the ``{{pii_type:value}}`` annotation marker."""
    return "{{" + pii_type + ":" + value + "}}"


def _tpl_bank_letter(f: PIIValueFactory) -> str:
    """Bank/utility letter: email, iban, italian_id, ip_address."""
    return (
        "Gentile cliente,\n\n"
        f"la contattiamo all'indirizzo {_ann('email', f.email())} in merito al "
        f"bonifico sull'IBAN {_ann('iban', f.iban())} (pratica n. {f.rng.randint(10000, 99999)}).\n"
        f"Per la fatturazione risulta il codice fiscale {_ann('italian_id', f.italian_id())}.\n"
        f"L'ultimo accesso al portale proviene dall'indirizzo IP "
        f"{_ann('ip_address', f.ip_address())}.\n\n"
        "Cordiali saluti,\nServizio Clienti"
    )


def _tpl_hr_record(f: PIIValueFactory) -> str:
    """HR onboarding record: person_name, date_of_birth, italian_id, address, phone, email."""
    return (
        "Scheda di assunzione\n\n"
        f"Dipendente: {_ann('person_name', f.person_name())}\n"
        f"Nato/a il: {_ann('date_of_birth', f.date_of_birth())}\n"
        f"Codice fiscale: {_ann('italian_id', f.italian_id())}\n"
        f"Residenza: {_ann('address', f.address())}\n"
        f"Recapito: {_ann('phone', f.phone())} — {_ann('email', f.email())}\n"
        f"Matricola interna: {f.rng.randint(1000, 9999)}"
    )


def _tpl_clinic_note(f: PIIValueFactory) -> str:
    """Clinical note: person_name, date_of_birth, health_data, italian_id, phone."""
    return (
        "Referto ambulatoriale\n\n"
        f"Paziente {_ann('person_name', f.person_name())}, "
        f"nato/a il {_ann('date_of_birth', f.date_of_birth())} "
        f"(CF {_ann('italian_id', f.italian_id())}).\n"
        f"Diagnosi: {_ann('health_data', f.health_data())}. "
        f"Si consiglia controllo tra sei mesi.\n"
        f"Per appuntamenti contattare il numero {_ann('phone', f.phone())}."
    )


def _tpl_support_ticket(f: PIIValueFactory) -> str:
    """Support ticket / access log: email, ip_address, credit_card, phone."""
    return (
        f"Ticket #{f.rng.randint(100000, 999999)} — pagamento non riuscito\n\n"
        f"Utente {_ann('email', f.email())} segnala l'addebito sulla carta "
        f"{_ann('credit_card', f.credit_card())} non andato a buon fine.\n"
        f"Tentativo registrato dall'IP {_ann('ip_address', f.ip_address())}. "
        f"Richiamare al {_ann('phone', f.phone())} in orario d'ufficio."
    )


def _tpl_cross_border(f: PIIValueFactory) -> str:
    """Swiss cross-border context: person_name, swiss_avs, iban, address."""
    return (
        "Notifica frontaliere\n\n"
        f"Il/la lavoratore/trice {_ann('person_name', f.person_name())} "
        f"(n. AVS {_ann('swiss_avs', f.swiss_avs())}) percepisce lo stipendio "
        f"sull'IBAN {_ann('iban', f.iban())}.\n"
        f"Domicilio dichiarato: {_ann('address', f.address())}."
    )


def _tpl_registry_rows(f: PIIValueFactory) -> str:
    """Spreadsheet-style export rows: person_name, email, phone, italian_id (+ distractors)."""
    lines = ["Estratto del registro dipendenti (export CSV):", ""]
    for _ in range(3):
        lines.append(
            f"{_ann('person_name', f.person_name())}; {_ann('email', f.email())}; "
            f"{_ann('phone', f.phone())}; CF {_ann('italian_id', f.italian_id())}; "
            f"matricola {f.rng.randint(1000, 9999)}"
        )
    return "\n".join(lines)


#: Document templates cycled to build the corpus; together they cover all
#: 11 ``pii_type`` categories of the catalog.
_TEMPLATES: tuple[Callable[[PIIValueFactory], str], ...] = (
    _tpl_bank_letter,
    _tpl_hr_record,
    _tpl_clinic_note,
    _tpl_support_ticket,
    _tpl_cross_border,
    _tpl_registry_rows,
)


def generate_documents(n: int, seed: int) -> list[tuple[str, str]]:
    """Generate ``n`` annotated documents, cycling the templates.

    :param n: number of documents to produce.
    :param seed: seed making the output reproducible.
    :returns: ``(document_id, annotated_text)`` pairs, in generation order.
    """
    factory = PIIValueFactory(seed)
    return [
        (f"gen_{i + 1:04d}", _TEMPLATES[i % len(_TEMPLATES)](factory))
        for i in range(n)
    ]


def default_generated_dir() -> Path:
    """:returns: the packaged ``pii_detection/evaluation/documents_generated`` path."""
    return Path(__file__).resolve().parent / "documents_generated"


def write_corpus(out_dir: Path, n: int, seed: int, *, emit_clean: bool = False) -> list[Path]:
    """Generate the corpus and write one annotated ``.txt`` per document.

    The annotated file is the single source. With ``emit_clean`` the marker-free
    text — exactly what a detector receives — is *derived* (via
    :func:`~pii_detection.evaluation.corpus.parse_annotated_text`, no re-stripping)
    into a ``clean/`` subdirectory, for inspection only. It is a build artifact,
    never a second source to edit: the loader's non-recursive ``*.txt`` glob does
    not pick it up.

    :param out_dir: destination directory for the annotated files (created if
        missing).
    :param n: number of documents to produce.
    :param seed: reproducibility seed.
    :param emit_clean: also write the derived clean text under ``out_dir/clean``.
    :returns: the annotated paths written, in generation order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = out_dir / "clean"
    if emit_clean:
        clean_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for doc_id, text in generate_documents(n, seed):
        path = out_dir / f"{doc_id}.txt"
        path.write_text(text, encoding="utf-8")
        written.append(path)
        if emit_clean:
            clean_text = parse_annotated_text(doc_id, text).text
            (clean_dir / f"{doc_id}.txt").write_text(clean_text, encoding="utf-8")
    return written


def validate_factory(seed: int, rounds: int = 200) -> None:
    """Assert the checksum-bearing values are valid, independently of Presidio.

    :param seed: reproducibility seed.
    :param rounds: how many values per category to check.
    :raises AssertionError: on the first value that fails its own checksum.
    """
    factory = PIIValueFactory(seed)
    for _ in range(rounds):
        assert _iban_is_valid(factory.iban())
        assert _luhn_is_valid(factory.credit_card())
        avs = factory.swiss_avs().replace(".", "")
        assert _ean13_check_digit(avs[:12]) == avs[12]
        assert codicefiscale.is_valid(factory.italian_id())


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: write the corpus to disk.

    :param argv: argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Generate the synthetic detection corpus.")
    parser.add_argument("--out", type=Path, default=default_generated_dir())
    parser.add_argument("--n", type=int, default=60, help="number of documents")
    parser.add_argument("--seed", type=int, default=42, help="reproducibility seed")
    parser.add_argument(
        "--emit-clean",
        action="store_true",
        help="also write derived marker-free text under <out>/clean (inspection only)",
    )
    args = parser.parse_args(argv)
    paths = write_corpus(args.out, args.n, args.seed, emit_clean=args.emit_clean)
    print(f"wrote {len(paths)} annotated documents to {args.out}")
    if args.emit_clean:
        print(f"wrote {len(paths)} clean documents to {args.out / 'clean'}")


if __name__ == "__main__":
    main()


__all__ = [
    "PIIValueFactory",
    "generate_documents",
    "write_corpus",
    "validate_factory",
    "default_generated_dir",
    "main",
]
