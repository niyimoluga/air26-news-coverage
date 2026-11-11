#!/usr/bin/env bash
# P1: filter 24 cached shards into data/processed/docs_YYYY-MM.parquet.
set -euo pipefail
cd "$(dirname "$0")/.."

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate air26

CACHE_DIR="${CACHE_DIR:-$HOME/infini-news/.cache/infini}"
python -m src.data.preprocess \
  --manifest data/manifest.csv \
  --cache-dir "$CACHE_DIR" \
  --out-dir data/processed \
  --download \
  --overwrite \
  "$@"
