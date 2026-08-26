#!/usr/bin/env bash
# Stato del benchmark AI notturno: a che punto è, e quanto manca.
#
# Il benchmark stampa i risultati solo alla fine, quindi l'avanzamento si ricava
# da due tracce indirette: le inizializzazioni di codecarbon nel file di output
# (una per riga misurata: baseline, AI da sola, fusione) e le richieste servite
# da Ollama (una per finestra di testo).
#
#   bash scripts/bench_status.sh
#
# CHUNKS = finestre di testo del campione (64 per --sample 20 --seed 42).
set -uo pipefail
cd "$(dirname "$0")/.."
CHUNKS="${CHUNKS:-64}"

echo "── modello in esecuzione ────────────────────────────────"
docker compose exec -T ollama ollama ps 2>/dev/null | tail -n +2 | grep -v '^$' \
  || echo "  (nessun modello caricato: fase senza AI, oppure finito)"

echo
echo "── avanzamento per modello ──────────────────────────────"
shopt -s nullglob
for f in doc/eval_results/04_ai_*.txt; do
  # salta il file della misura precedente, superata dal corpus aziendale
  [ "$(basename "$f")" = "04_ai_benchmark.txt" ] && continue
  phases=$(grep -c "offline tracker init" "$f" 2>/dev/null || true)
  finished=$(grep -c "^OVERALL" "$f" 2>/dev/null || true)
  if [ "${finished:-0}" -gt 0 ]; then stato="COMPLETATO"
  else
    case "${phases:-0}" in
      0) stato="avvio (caricamento detector)" ;;
      1) stato="1/3 · baseline pattern+NER" ;;
      2) stato="2/3 · AI da sola" ;;
      3) stato="3/3 · fusione tradizionale+AI" ;;
      *) stato="${phases} fasi avviate" ;;
    esac
  fi
  printf '  %-28s %s\n' "$(basename "$f")" "$stato"
done

# Le richieste vanno contate solo da quando il run corrente e' partito, altrimenti
# il totale include quelle dei run precedenti. L'istante di avvio si ricava dalla
# prima inizializzazione di codecarbon nel file piu' recente (ora del container).
newest=$(ls -t doc/eval_results/04_ai_*.txt 2>/dev/null | grep -v 04_ai_benchmark.txt | head -1)
since=""
if [ -n "${newest:-}" ]; then
  t=$(grep -m1 -o "offline tracker init" -B0 "$newest" >/dev/null 2>&1 && \
      grep -m1 "offline tracker init" "$newest" | sed -E 's/.*@ ([0-9:]+).*/\1/')
  day=$(docker compose exec -T ollama date -u '+%Y-%m-%d' 2>/dev/null | tr -d '\r')
  [ -n "$t" ] && [ -n "$day" ] && since="${day}T${t}Z"
fi

if [ -n "$since" ]; then
  served=$(docker compose logs ollama --since "$since" 2>/dev/null | grep -c 'POST     "/api/chat"' || true)
  rate=$(docker compose logs ollama --since 10m 2>/dev/null | grep -c 'POST     "/api/chat"' || true)
else
  served=$(docker compose logs ollama 2>/dev/null | grep -c 'POST     "/api/chat"' || true)
  rate=0
fi

total=$((CHUNKS * 2))
echo
echo "── richieste all'AI (run corrente) ──────────────────────"
printf '  %s / %s finestre\n' "${served:-0}" "$total"
if [ "${rate:-0}" -gt 0 ]; then
  left=$(( total - ${served:-0} ))
  [ "$left" -lt 0 ] && left=0
  eta=$(( left * 10 / rate ))
  printf '  ritmo: %s ogni 10 min  ·  mancano ~%s min\n' "$rate" "$eta"
else
  echo "  ritmo: nessuna richiesta negli ultimi 10 min (fase senza AI, o finito)"
fi
