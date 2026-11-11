#!/usr/bin/env bash
# P3 helper: list seed clusters for query authoring.
set -euo pipefail
cd "$(dirname "$0")/.."

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate air26

python -m src.queries.select_seeds \
  --clusters-dir data/processed \
  --docs-dir data/processed \
  --out queries/seed_candidates.csv \
  "$@"
