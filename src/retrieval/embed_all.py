"""Precompute MiniLM embeddings for every month in the manifest.

`run_all` encodes a month lazily when a query needs it. This script does it
ahead of time so the first 60-query run does not spend 15 minutes on the GPU.
Embeddings are cached in `data/processed/emb_YYYY-MM.npy` with an id list; the
cache is reused as long as the document order is unchanged.

Usage:
    python -m src.retrieval.embed_all --months 2023-01,2023-02
"""

from __future__ import annotations

import argparse
import time

import torch

from ..data.common import load_config, load_manifest, resolve, select_months
from .dense import DEFAULT_MODEL, load_model
from .run_all import embedding_cache, load_month


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--months", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    docs_dir = resolve(args.docs_dir or cfg["paths"]["docs_dir"])
    months = select_months(load_manifest(args.manifest), args.months)["month"].tolist()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device=device)
    print(f"[env] device={device} model={args.model} months={len(months)}")

    started = time.perf_counter()
    for i, month in enumerate(months, 1):
        docs = load_month(docs_dir, month)
        texts = (
            docs["title"].fillna("").astype(str)
            + " "
            + docs["text"].fillna("").astype(str).str.slice(0, 2000)
        ).tolist()
        t0 = time.perf_counter()
        emb = embedding_cache(
            month, docs, texts, docs_dir, model,
            overwrite=args.overwrite, batch_size=args.batch_size,
        )
        print(f"[{i}/{len(months)}] {month} {emb.shape} in {time.perf_counter() - t0:.1f}s")

    print(f"[done] {len(months)} months in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
