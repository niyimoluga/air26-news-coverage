"""S1/S2 retrieval over month-scoped pools.

Each query is retrieved only from the documents of its own month. The script
asserts this, saves top-k runs, latency measurements and a seed-cluster sanity
check. It does not read the qrels.

Usage:
    python -m src.retrieval.run_all --queries queries/queries.csv \
        --systems bm25,dense --top-k 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.common import load_config, resolve
from .bm25 import BM25Retriever
from .dense import DEFAULT_MODEL, DenseRetriever, encode_documents, load_model


def load_queries(path: str | Path, status: str | None, months: str | None) -> pd.DataFrame:
    queries = pd.read_csv(resolve(path), dtype={"month": str, "seed_cluster": str})
    required = {"qid", "text", "month"}
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"queries file missing columns: {sorted(missing)}")
    if "seed_cluster" not in queries.columns:
        queries["seed_cluster"] = None
    queries = queries.dropna(subset=["qid", "text", "month"]).copy()
    if status and "status" in queries.columns:
        wanted = {s.strip() for s in status.split(",") if s.strip()}
        queries = queries[queries["status"].isin(wanted)]
    if months:
        wanted = {m.strip() for m in months.split(",") if m.strip()}
        queries = queries[queries["month"].isin(wanted)]
    return queries.reset_index(drop=True)


def load_month(docs_dir: Path, month: str) -> pd.DataFrame:
    path = docs_dir / f"docs_{month}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run P1 first")
    docs = pd.read_parquet(
        path, columns=["doc_id", "month", "cluster_id", "title", "text"]
    )
    if (docs["month"] != month).any():
        raise ValueError(f"{path} contains documents from another month")
    return docs


def embedding_cache(
    month: str,
    docs: pd.DataFrame,
    texts: list[str],
    docs_dir: Path,
    model,
    *,
    overwrite: bool,
    batch_size: int,
) -> np.ndarray:
    emb_path = docs_dir / f"emb_{month}.npy"
    ids_path = docs_dir / f"emb_ids_{month}.json"
    doc_ids = docs["doc_id"].astype(str).tolist()
    if emb_path.exists() and ids_path.exists() and not overwrite:
        cached_ids = json.loads(ids_path.read_text())
        if cached_ids != doc_ids:
            raise ValueError(f"{ids_path} does not match docs order; rerun with --overwrite-embeddings")
        return np.load(emb_path)
    embeddings = encode_documents(model, texts, batch_size=batch_size)
    np.save(emb_path, embeddings)
    ids_path.write_text(json.dumps(doc_ids))
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="queries/queries.csv")
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--systems", default="bm25,dense")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--status", default=None,
                        help="comma-separated status filter, e.g. draft or pilot")
    parser.add_argument("--months", default=None)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite-embeddings", action="store_true")
    parser.add_argument("--min-seed-hit", type=float, default=0.8)
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if a system's seed hit rate is below --min-seed-hit")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    docs_dir = resolve(args.docs_dir or cfg["paths"]["docs_dir"])
    out_dir = resolve(args.out_dir or cfg["paths"]["runs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.queries, args.status, args.months)
    if args.limit_queries:
        queries = queries.head(args.limit_queries)
    if queries.empty:
        raise SystemExit("no queries selected")

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - {"bm25", "dense"}
    if unknown:
        raise ValueError(f"unknown systems: {sorted(unknown)}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device=device) if "dense" in systems else None
    print(f"[env] device={device} systems={systems} queries={len(queries)}")

    runs: dict[str, list[pd.DataFrame]] = {s: [] for s in systems}
    latency_rows: list[dict] = []
    seed_rows: list[dict] = []

    for month, month_queries in queries.groupby("month", sort=True):
        docs = load_month(docs_dir, month)
        doc_ids = docs["doc_id"].astype(str).tolist()
        doc_id_set = set(doc_ids)
        cluster_of = dict(zip(doc_ids, docs["cluster_id"].astype(str)))
        texts = (
            docs["title"].fillna("").astype(str)
            + " "
            + docs["text"].fillna("").astype(str).str.slice(0, 2000)
        ).tolist()

        retrievers: dict[str, object] = {}
        if "bm25" in systems:
            started = time.perf_counter()
            retrievers["bm25"] = BM25Retriever(texts, doc_ids)
            print(f"[{month}] bm25 index: {len(doc_ids):,} docs in {time.perf_counter() - started:.1f}s")
        if "dense" in systems:
            embeddings = embedding_cache(
                month, docs, texts, docs_dir, model,
                overwrite=args.overwrite_embeddings, batch_size=args.batch_size,
            )
            retrievers["dense"] = DenseRetriever(model, doc_ids, embeddings)
            print(f"[{month}] dense index: {embeddings.shape}")

        for query in month_queries.itertuples(index=False):
            for system in systems:
                retriever = retrievers[system]
                timings = []
                result = None
                for _ in range(max(1, args.latency_repeats)):
                    started = time.perf_counter()
                    result = retriever.search(query.text, k=args.top_k)
                    timings.append((time.perf_counter() - started) * 1000.0)
                assert result is not None

                result = result.reset_index(drop=True)
                result["rank"] = np.arange(1, len(result) + 1)
                if not set(result["doc_id"]).issubset(doc_id_set):
                    raise AssertionError(f"{query.qid}/{system}: retrieved docs outside {month}")
                result["qid"] = query.qid
                result["system"] = system
                result["month"] = month
                runs[system].append(result[["qid", "system", "month", "doc_id", "rank", "score"]])

                latency_rows.append(
                    {
                        "qid": query.qid,
                        "system": system,
                        "month": month,
                        "median_ms": float(np.median(timings)),
                        "repeats": len(timings),
                        "docs_in_pool": len(doc_ids),
                    }
                )

                seed_cluster = getattr(query, "seed_cluster", None)
                if pd.notna(seed_cluster) and str(seed_cluster).strip():
                    ranks = [
                        int(rank)
                        for doc_id, rank in zip(result["doc_id"], result["rank"])
                        if cluster_of.get(doc_id) == str(seed_cluster)
                    ]
                    seed_rows.append(
                        {
                            "qid": query.qid,
                            "system": system,
                            "month": month,
                            "seed_cluster": str(seed_cluster),
                            "seed_hit": bool(ranks),
                            "seed_rank": min(ranks) if ranks else None,
                        }
                    )

    for system in systems:
        if not runs[system]:
            continue
        frame = pd.concat(runs[system], ignore_index=True)
        path = out_dir / f"{system}__none.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        print(f"[save] {len(frame):,} rows -> {path}")

    latency = pd.DataFrame(latency_rows)
    latency.to_csv(out_dir / "latency.csv", index=False)
    print(f"[save] latency -> {out_dir / 'latency.csv'}")

    if seed_rows:
        seeds = pd.DataFrame(seed_rows)
        seeds.to_csv(out_dir / "seed_check.csv", index=False)
        summary = seeds.groupby("system")["seed_hit"].mean().sort_index()
        print("\n[seed check] share of pilot queries whose seed cluster is in top-k:")
        for system, rate in summary.items():
            print(f"  {system:6s} {rate:.1%}")
        low = summary[summary < args.min_seed_hit]
        if len(low) and args.strict:
            raise SystemExit(f"seed hit rate below {args.min_seed_hit:.0%}: {low.to_dict()}")
    else:
        seeds = pd.DataFrame()
        print("[seed check] no seed_cluster column/values; skipped")

    if not latency.empty:
        print("\n[latency] median of per-query medians (ms):")
        print(latency.groupby("system")["median_ms"].median().round(1).to_string())


if __name__ == "__main__":
    main()
