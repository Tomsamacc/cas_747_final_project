#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OUT="${ROOT}/results/FINAL_RESULTS.txt"
mkdir -p "${ROOT}/results"
{
  echo "======== run_train_and_results.sh $(date +'%Y-%m-%dT%H:%M:%S') ========"
  echo ""
} | tee "$OUT"

train_and_metric() {
  local ds=$1
  local n=$2
  echo "" | tee -a "$OUT"
  echo "=== train $ds ($n split(s): indices 0..$((n - 1))) ===" | tee -a "$OUT"
  local i
  for ((i = 0; i < n; i++)); do
    python -m src.models.train --dataset "$ds" --processed-name "$ds" --split-idx "$i" || {
      echo "FAILED train $ds split $i" | tee -a "$OUT"
    }
  done
  python -m src.utils.cal_mean_metric --dataset "$ds" --last-n "$n" 2>&1 | tee -a "$OUT" || {
    echo "FAILED cal_mean_metric $ds" | tee -a "$OUT"
  }
}

while IFS=: read -r ds n; do
  [[ -z "${ds// }" ]] && continue
  [[ "$ds" =~ ^# ]] && continue
  train_and_metric "$ds" "$n"
done <<'EOF'
cora:10
citeseer:10
pubmed:10
chameleon:10
squirrel:10
actor:10
cornell:10
texas:10
wisconsin:10
ogbn-arxiv:1
arxiv-year:5
EOF

{
  echo ""
  echo "======== end $(date +'%Y-%m-%dT%H:%M:%S') ========"
  echo "Full log saved to: $OUT"
} | tee -a "$OUT"
