#!/usr/bin/env bash
# P4: BM25 + dense retrieval on the pilot queries (month-scoped).
set -euo pipefail
cd "$(dirname "$0")/.."

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate air26

python -m src.retrieval.run_all \
  --queries queries/queries.csv \
  --docs-dir data/processed \
  --out-dir results/runs \
  --systems bm25,dense \
  --top-k 100 \
  --latency-repeats 5 \
  "$@"
