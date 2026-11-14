#!/usr/bin/env bash
# P5: build the pilot candidate pool and the annotation sheets.
set -euo pipefail
cd "$(dirname "$0")/.."

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate air26

python -m src.queries.build_candidates \
  --queries queries/queries.csv \
  --runs-dir results/runs \
  --docs-dir data/processed \
  --systems bm25,dense \
  --per-system 10 \
  --cap 15 \
  --out queries/candidates.csv \
  "$@"

python -m src.queries.make_sheets \
  --mode init \
  --candidates queries/candidates.csv \
  --out-dir queries \
  --force
