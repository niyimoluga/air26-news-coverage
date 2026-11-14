#!/usr/bin/env bash
# P5: build the pilot candidate pool and the annotation sheets.
# Refuses to overwrite existing annotations unless FORCE=1.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f queries/candidates.csv ] && [ "${FORCE:-0}" != "1" ]; then
  echo "queries/candidates.csv already exists; refusing to overwrite annotations."
  echo "Set FORCE=1 to rebuild the pool and reset the sheets."
  exit 1
fi

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

FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAG="--force"; fi
python -m src.queries.make_sheets \
  --mode init \
  --candidates queries/candidates.csv \
  --out-dir queries \
  $FORCE_FLAG
