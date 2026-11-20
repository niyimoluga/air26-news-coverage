#!/usr/bin/env bash
# P6: S1-S4 retrieval on all active queries over all 24 monthly pools.
# Run this only after queries.csv contains the 50 evaluation queries.
set -euo pipefail
cd "$(dirname "$0")/.."

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate air26

python -m src.retrieval.run_all \
  --queries queries/queries.csv \
  --docs-dir data/processed \
  --out-dir results/runs \
  --systems bm25,dense,crossencoder,hybrid \
  --status pilot,eval \
  --top-k 100 \
  --latency-repeats 5 \
  "$@"
