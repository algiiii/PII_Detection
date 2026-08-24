#!/usr/bin/env bash
# =============================================================================
# thesis_eval.sh — produce ogni tabella del Capitolo 8 (Test e validazione).
#
# Esegue, nell'ordine del capitolo, tutti i test/valutazioni descritti e salva
# ciascun output in $OUTDIR, così da poter copiare i numeri nelle tabelle LaTeX.
#
# DOVE ESEGUIRLO
#   La ML pesante (GLiNER) e i modelli generativi (Ollama) girano SOLO nel
#   container. Il modo canonico è quindi:
#       docker compose up -d --build
#       docker compose exec app bash scripts/thesis_eval.sh
#   In locale (host x86/Rosetta) girano solo i test di unità e il livello a
#   pattern: lanciare con  GLINER=0  SKIP_AI=1  per saltare ciò che serve il container.
#
# KNOBS (variabili d'ambiente)
#   PY        interprete Python           (default: ./.venv/bin/python)
#   OUTDIR    cartella degli output       (default: doc/eval_results)
#   GLINER    1=usa GLiNER per il NER      (default: 1; metti 0 per spaCy in locale)
#   MODELS    modelli AI da confrontare   (default: lista a tre fasce di taglia)
#   LIMIT     valuta solo i primi N doc   (default: vuoto = tutto il corpus)
#   CORPUS    cartella del corpus annotato (default: quello generato, 60 doc /
#             350 occorrenze; metti "" per il corpus piccolo impacchettato)
#   SKIP_UNIT / SKIP_DETECT / SKIP_PIPELINE / SKIP_AI = 1 per saltare una fase
#
# ESEMPI
#   # esplorativo e veloce (pochi doc, un solo modello):
#   LIMIT=10 MODELS=phi4-mini docker compose exec app bash scripts/thesis_eval.sh
#   # completo:
#   docker compose exec app bash scripts/thesis_eval.sh
# =============================================================================
set -euo pipefail

# --- posizionati nella radice del progetto (lo script sta in scripts/) --------
cd "$(dirname "$0")/.."

PY="${PY:-./.venv/bin/python}"
OUTDIR="${OUTDIR:-doc/eval_results}"
GLINER="${GLINER:-1}"
MODELS="${MODELS:-phi4-mini qwen3:4b gemma3:4b qwen2.5:7b gemma3:12b}"
LIMIT="${LIMIT:-}"
# Il corpus di riferimento del capitolo: 60 documenti / 350 occorrenze annotate,
# lo stesso da cui `render` produce i PDF della tassa di estrazione — così tutte
# le tabelle poggiano sullo stesso terreno.
CORPUS="${CORPUS-pii_detection/evaluation/documents_generated}"

# nel container l'interprete di sistema va bene se non c'è il venv
if [ ! -x "$PY" ]; then PY="python"; fi

mkdir -p "$OUTDIR"

# flag condivisi
gliner_flag=""; [ "$GLINER" = "1" ] && gliner_flag="--gliner"
limit_flag="";  [ -n "$LIMIT" ]    && limit_flag="--limit $LIMIT"
corpus_flag=""; [ -n "$CORPUS" ]   && corpus_flag="--corpus $CORPUS"

hr() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# =============================================================================
# §8.1 — Verifica funzionale: i test di unità   -> tab:test-unit
# =============================================================================
if [ "${SKIP_UNIT:-0}" != "1" ]; then
  hr "§8.1  Test di unità (pytest)  ->  tab:test-unit"
  # -q riepiloga; --co conta i casi per area (per riempire la colonna 'N. casi').
  { $PY -m pytest -q; echo; echo "--- conteggio casi per area ---"; \
    $PY -m pytest --collect-only -q 2>/dev/null | grep -c "::test_" || true; } \
    | tee "$OUTDIR/01_unit_tests.txt"
fi

# =============================================================================
# §8.3.1/§8.3.2 (tradizionale) — Presidio pattern, NER, union, per categoria
#   -> tab:test-pattern, tab:test-ner, e le righe "tradizionali" di
#      tab:test-detection-confronto
# =============================================================================
if [ "${SKIP_DETECT:-0}" != "1" ]; then
  hr "§8.3.1  Presidio pattern / NER / union (per categoria)  ->  tab:test-pattern, tab:test-ner"
  # Stampa tre report per-categoria: pattern da solo, NER da solo, union.
  # shellcheck disable=SC2086
  $PY -m pii_detection.evaluation.run_presidio_baseline $gliner_flag $corpus_flag \
    | tee "$OUTDIR/02_presidio_per_category.txt"
fi

# =============================================================================
# §8.3.3 (tradizionale) — Sistema tradizionale end-to-end + "tassa di estrazione"
#   -> righe tradizionali di tab:test-sistema-intero (Tier 1 = testo pulito)
#      e misura del costo dell'estrazione (Tier 2)
# =============================================================================
if [ "${SKIP_PIPELINE:-0}" != "1" ]; then
  hr "§8.3.3  Pipeline end-to-end (estrai->detect->score) + tassa di estrazione"
  # 1) rende il corpus sintetico in PDF/DOCX (idempotente) ...
  $PY -m pii_detection.evaluation.render >/dev/null
  # 2) ... poi valuta pattern+NER sul testo pulito e su quello estratto.
  $PY -m pii_detection.evaluation.run_pipeline $gliner_flag \
    | tee "$OUTDIR/03_pipeline_extraction_tax.txt"
fi

# =============================================================================
# §8.3.1 (AI) + §8.3.2 (agentico) + §8.3.3 (con AI) + §8.4 — Benchmark AI multi-modello
#   -> tab:test-ai (AI per categoria), righe "+ AI" di tab:test-detection-confronto,
#      tab:test-sistema-intero (con AI), e tab:test-ai-benchmark (qualità vs costo)
#   Richiede Ollama con i modelli scaricati (container).
# =============================================================================
if [ "${SKIP_AI:-0}" != "1" ]; then
  hr "§8.4  Benchmark AI multi-modello (qualità vs costo) + per-categoria  ->  tab:test-ai, tab:test-ai-benchmark"
  # --per-category espande ogni riga (ai:<m>, union:<m>) nel dettaglio per pii_type;
  # la tabella compatta in testa dà P/R/F1 + s/doc + Wh/doc per il confronto di taglia.
  # shellcheck disable=SC2086
  $PY -m pii_detection.evaluation.run_ai_benchmark $gliner_flag $limit_flag $corpus_flag \
      --models $MODELS --per-category \
    | tee "$OUTDIR/04_ai_benchmark.txt"
fi

hr "Fatto. Output salvati in $OUTDIR/"
ls -1 "$OUTDIR"
