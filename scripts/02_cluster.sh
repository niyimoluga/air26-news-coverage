#!/usr/bin/env bash
# P2: per-month exact + MinHash/LSH story clustering.
set -euo pipefail
cd "$(dirname "$0")/.."

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate air26

python -m src.data.cluster \
  --docs-dir data/processed \
  --out-dir data/processed \
  --jobs "${JOBS:-8}" \
  --overwrite \
  "$@"
