#!/usr/bin/env bash
# P6 helper: richness-ranked seed shortlist for the 50 evaluation queries.
# Ranks by the number of distinct headlines covering the same event, not by
# raw cluster size (the pilot showed size is inflated by wire fragmentation).
set -euo pipefail
cd "$(dirname "$0")/.."

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate air26

python -m src.queries.select_seeds \
  --clusters-dir data/processed \
  --docs-dir data/processed \
  --out queries/seed_shortlist.csv \
  --richness \
  --min-richness 1 \
  --per-month 8 \
  "$@"
