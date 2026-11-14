#!/usr/bin/env bash
# P5: validate the pilot judgments and compute Cohen's kappa.
set -euo pipefail
cd "$(dirname "$0")/.."

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate air26

python -m src.queries.validate_judgments \
  --candidates queries/candidates.csv \
  --judgments queries/judgments.csv \
  --require-complete

python -m src.queries.validate_judgments \
  --candidates queries/candidates.csv \
  --judgments queries/judgments_double.csv

python -m src.queries.kappa \
  --judgments queries/judgments.csv \
  --double queries/judgments_double.csv
