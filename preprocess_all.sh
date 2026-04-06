#!/usr/bin/env bash
cd "$(dirname "$0")"
for ds in cora citeseer pubmed chameleon squirrel cornell texas wisconsin actor; do
  echo "=== $ds ==="
  python -m src.data_processing.preprocess --dataset "$ds" || echo "FAILED: $ds"
done
