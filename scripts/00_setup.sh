#!/usr/bin/env bash
# P0: verify the environment and create the output folders.
set -euo pipefail
cd "$(dirname "$0")/.."

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate air26

python - <<'PY'
import torch, sentence_transformers, bm25s, datasketch, ir_measures
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("sentence-transformers", sentence_transformers.__version__)
print("bm25s", bm25s.__version__)
print("datasketch", datasketch.__version__)
print("ir_measures", ir_measures.__version__)
PY

mkdir -p data/processed results/runs results/metrics results/figures
echo "repo: $(pwd)"
echo "manifest rows: $(($(wc -l < data/manifest.csv) - 1))"
echo "setup OK"
