#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DO_INSTALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)
      DO_INSTALL=1
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [--install]"
      echo "  --install  pip install -U pip && pip install -r requirements.txt"
      echo "  (no args)  only run preprocessing"
      echo "Activate your environment first; install PyTorch + PyG per README before --install if using GPU."
      exit 0
      ;;
    *)
      echo "unknown option: $1 (try --help)"
      exit 1
      ;;
  esac
done

if [[ "$DO_INSTALL" -eq 1 ]]; then
  echo "==> pip install -r requirements.txt (PyTorch + PyG should already match your machine per README)"
  python -m pip install -U pip
  python -m pip install -r requirements.txt
fi

echo "==> preprocess small graphs (preprocess_all.sh)"
bash "$ROOT/preprocess_all.sh"

echo "==> preprocess ogbn-arxiv"
python -m src.data_processing.preprocess --dataset ogbn-arxiv || {
  echo "FAILED: ogbn-arxiv (check data/raw / network / ogb)"
}

echo "==> preprocess arxiv-year"
python -m src.data_processing.preprocess --dataset arxiv-year || {
  echo "FAILED: arxiv-year"
}

echo "==> done setup_and_preprocess.sh (processed files under data/processed/)"
