# PII Discovery

Rilevamento di dati personali (PII) in documenti aziendali non strutturati e
**verifica di conformità** rispetto al registro dei trattamenti dichiarato dal
titolare (ROPA), secondo nLPD e GDPR.

> Prototipo realizzato come lavoro di tesi di Bachelor in Ingegneria Informatica
> (SUPSI, 2025–2026). Il documento completo — analisi, architettura, misure
> sperimentali — è in [`doc/DOC_Algisi.pdf`](doc/DOC_Algisi.pdf).

## Il problema

Un'organizzazione dichiara nel proprio ROPA *quali* categorie di dati personali
tratta, per quali finalità e per quanto tempo le conserva. Ciò che sta davvero nei
suoi documenti — contratti, verbali, fogli di calcolo sparsi in cartelle condivise —
nessuno lo sa. Fra il registro e i file esiste uno scarto che oggi si misura solo a
mano.

Gli strumenti liberi (Presidio, GLiNER) sanno **trovare** le PII in un testo, ma si
fermano lì; le suite commerciali coprono l'intera catena a costi e condizioni di
opacità che una PMI non sostiene. Questo progetto occupa la casella vacante: trovare
le PII **e** ricondurle a ciò che il registro dichiara, con strumenti ispezionabili
che girano interamente in casa.

## Cosa fa

```
documenti  ──►  estrazione  ──►  rilevamento ibrido  ──►  registro PII
(PDF/DOCX/                       pattern + NER              (tipi e posizioni,
 XLSX/ODS/txt)                   + AI opzionale              mai i valori)
                                                                  │
      ROPA (.ods/.xlsx) ──► categorie dichiarate ──► confronto ◄──┘
                                                        │
                                            verdetto: PII non dichiarate,
                                            categorie mancanti, retention scaduta
```

| Blocco | Funzionalità |
|--------|--------------|
| **B1** | Ingestione del ROPA da template CNIL (`.ods`/`.xlsx`) e mappatura delle categorie dichiarate sul catalogo interno |
| **B3** | Estrazione e normalizzazione del testo (born-digital) + data di riferimento del documento |
| **B4** | Rilevamento ibrido: pattern con checksum, NER zero-shot (GLiNER), secondo parere generativo opzionale, fusi da un merge engine |
| **B5** | Registro delle PII rilevate, sincronizzato col disco a ogni scansione |
| **B6** | Associazione documento→trattamento, manuale o per regole cartella→trattamento |
| **B7** | Verifica di conformità: categorie non dichiarate, dichiarate e non trovate, termini di conservazione superati |
| **B8** | Interfaccia web per il DPO: dashboard, scansione, regole, retention, review del ROPA |

Tre proprietà governano il progetto:

- **Minimizzazione** — il registro conserva il *tipo* e la *posizione* di una PII, mai
  il suo valore. Uno strumento di tutela non deve diventare esso stesso un archivio di
  dati personali.
- **Tutto in locale** — nessun servizio cloud, nessuna chiamata verso l'esterno. Anche
  l'inferenza generativa gira in casa (Ollama, modelli piccoli quantizzati su CPU).
- **Containerizzato** — ogni componente pesante è un servizio su porta configurato da
  variabili d'ambiente; la stessa immagine gira in locale e in produzione.

## Avvio rapido

Serve solo Docker.

```bash
docker compose up -d --build
docker compose exec ollama ollama pull phi4-mini   # una volta, resta nel volume
```

L'interfaccia è su **http://localhost:8000**. Da lì: `/ropa` per importare il registro
dei trattamenti, `/scan` per analizzare una cartella, `/` per la dashboard di
conformità.

Le CLI restano disponibili nello stesso container:

```bash
docker compose exec app python -m pii_detection.scan documento.pdf
docker compose exec app pytest -q
```

## Interfaccia web

| Pagina | A cosa serve |
|--------|--------------|
| `/` | Dashboard: documenti analizzati, PII per tipo, stato di conformità |
| `/document/{id}` | Scheda documento: PII rilevate, trattamenti associati, verdetto |
| `/scan` | Avvio di una scansione da percorso lato server o da upload, con anteprima e quota di analisi AI |
| `/rules` | Regole cartella→trattamento: una mappatura vale per tutti i file sotto quel prefisso |
| `/retention` | Tutto ciò che risulta conservato oltre il termine dichiarato, per gravità |
| `/ropa` | Review del registro: import del file e conferma delle mappature di categoria |

Il registro fornito dall'organizzazione è **fonte autoritativa**: il sistema lo importa
e lo confronta, non lo modifica mai.

## Riga di comando

```bash
# ROPA: importa un registro CNIL e mappa le categorie
python -m pii_detection.ropa.ingestion registro.ods

# Rilevamento su un singolo documento (a schermo, nulla persistito)
python -m pii_detection.scan documento.pdf

# Testo normalizzato che il rilevamento riceve (B3)
python -m pii_detection.extraction documento.pdf

# Scansione ricorsiva di una cartella nel registro PII
python -m pii_detection.registry.scan_folder /percorso/cartella [--gliner] [--ai] [--full]

# Conformità: associazione e verdetto
python -m pii_detection.compliance assign  <document_id> --activities id1,id2
python -m pii_detection.compliance check   <document_id>
python -m pii_detection.compliance retention
```

## Configurazione

Nessun host, percorso o nome di modello è cablato nel codice: tutto passa da variabili
d'ambiente (impostate in `docker-compose.yml`).

| Variabile | Default | Significato |
|-----------|---------|-------------|
| `ROPA_DB_URL` | `sqlite:///data/ropa.db` | Database del registro dei trattamenti |
| `PII_DB_URL` | `sqlite:///data/pii.db` | Database delle PII rilevate |
| `OLLAMA_HOST` | `http://ollama:11434` | Runtime LLM locale |
| `ROPA_LLM_MODEL` | `phi4-mini` | Modello per la mappatura delle categorie (B1) |
| `PII_LLM_MODEL` | `phi4-mini` | Modello per il rilevamento generativo (B4) |
| `PII_AI_SAMPLING_RATE` | `0` | Quota di documenti analizzati anche dall'AI: `0` mai, `1` tutti, `N` uno ogni `N` |
| `PII_LLM_NUM_PREDICT` | `1024` | Tetto ai token generati per richiesta |

## Estendere il rilevamento senza scrivere codice

Le regole di rilevamento sono dichiarative. Aggiungere un pattern è **una riga YAML** in
`pii_detection/config/custom_patterns.yaml`; se introduce una categoria nuova, una riga
in `categories.yaml`. Nessuna modifica al codice, nessuna ricompilazione.

## Sviluppo in locale

Il venv locale serve all'iterazione veloce sulla parte leggera; la ML pesante
(torch/GLiNER/spaCy) vive nel container.

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -e ".[ropa,registry,review,presidio,extraction,eval,llm,docs,dev]"
./.venv/bin/python -m spacy download it_core_news_lg   # tokenizzatore usato da Presidio

./.venv/bin/mypy && ./.venv/bin/ruff check pii_detection && ./.venv/bin/pytest -q
```

La reference tecnica delle API è generata da Sphinx a partire dai docstring:

```bash
./.venv/bin/sphinx-build -W -b html pii_detection/docs pii_detection/docs/_build/html
```

## Valutazione

Le misure riportate in tesi non si raccolgono a mano. I corpus sono **sintetici e
riproducibili da seed** — nessuna PII reale è mai stata trattata — con valori validi al
checksum (IBAN mod-97, Luhn, AVS, codice fiscale) e una quota deliberata di documenti
*privi* di PII, senza i quali i falsi positivi non sarebbero osservabili.

```bash
docker compose exec app bash scripts/thesis_eval.sh
```

Lo script esegue in sequenza la suite di unità, la generazione dei corpus e le passate
di valutazione, salvando l'esito di ciascuna in [`doc/eval_results/`](doc/eval_results) —
i file da cui le tabelle del Capitolo 8 sono trascritte.

## Struttura del repository

```
pii_detection/
├── config/       # cataloghi e regole in YAML (unica fonte dei pii_type)
├── extraction/   # B3 — lettura e normalizzazione dei documenti
├── detection/    # B4 — pattern, NER, rilevatore AI, merge engine
├── registry/     # B5 — registro delle PII rilevate, scansione di cartelle
├── ropa/         # B1 — ingestione del registro dei trattamenti + review
├── compliance/   # B6/B7 — associazione e verdetto
├── web/          # B8 — interfaccia del DPO
├── evaluation/   # corpus sintetici, benchmark, runner di misura
└── docs/         # sorgenti Sphinx della reference tecnica
doc/              # tesi LaTeX e risultati sperimentali
scripts/          # orchestrazione delle misure
```

## Limiti noti

- **Solo documenti nativi digitali**: nessun OCR, quindi le scansioni non sono coperte.
- **Un solo profilo utente** (il DPO) e **nessuna autenticazione**: strumento interno.
- **Web app a singolo worker**: i job di scansione vivono in memoria di processo.
- L'ingestione del ROPA segue il **template CNIL**, di cui legge la sola sezione
  «Categories of personal data».

Le funzionalità concepite e non realizzate, con il motivo per cui sono state escluse,
sono documentate nel Capitolo 10 della tesi.

## Licenza

Da definire prima della pubblicazione: senza un file `LICENSE` il codice è visibile ma
non riutilizzabile da terzi.
