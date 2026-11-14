"""P5: build the pilot candidate pool from S1/S2 runs.

For each query, walk each system's ranked list and collect the first
`--per-system` distinct story clusters; union them, rank by best rank across
systems, and cap at `--cap`. The seed cluster is always included so the event
the query was written from can be judged.

Output (gitignored because it contains titles/snippets):
    queries/candidates.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..data.common import load_config, resolve


def clean(text: str) -> str:
    return " ".join((text or "").split())


def load_month_assets(docs_dir: Path, month: str):
    docs = pd.read_parquet(
        docs_dir / f"docs_{month}.parquet",
        columns=["doc_id", "host", "text", "cluster_id"],
    )
    clusters = pd.read_parquet(
        docs_dir / f"clusters_{month}.parquet",
        columns=[
            "cluster_id",
            "size",
            "hosts",
            "earliest_publish",
            "earliest_warc",
            "rep_doc_id",
            "rep_title",
        ],
    ).set_index("cluster_id")
    return docs, clusters


def build_for_query(
    query,
    runs: dict[str, pd.DataFrame],
    docs: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    per_system: int,
    cap: int,
    snippet_chars: int,
) -> list[dict]:
    cluster_of = dict(zip(docs["doc_id"].astype(str), docs["cluster_id"].astype(str)))
    doc_meta = docs.set_index("doc_id")[["host", "text"]]

    best_rank: dict[str, int] = {}
    systems_for: dict[str, set[str]] = defaultdict(set)
    cluster_docs: dict[str, list[tuple[int, str, str]]] = defaultdict(list)

    for system, frame in runs.items():
        sub = frame[frame["qid"] == query.qid].sort_values("rank")
        seen: set[str] = set()
        for row in sub.itertuples(index=False):
            cid = cluster_of.get(str(row.doc_id))
            if not cid or cid == "nan":
                continue
            cluster_docs[cid].append((int(row.rank), str(row.doc_id), system))
            if cid in seen:
                continue
            seen.add(cid)
            systems_for[cid].add(system)
            best_rank[cid] = min(best_rank.get(cid, 10**9), int(row.rank))
            if len(seen) >= per_system:
                break

    seed = str(getattr(query, "seed_cluster", "") or "")
    seed_forced = bool(seed and seed not in best_rank)
    if seed_forced:
        seed_ranks: list[int] = []
        for system, frame in runs.items():
            sub = frame[frame["qid"] == query.qid]
            for row in sub.itertuples(index=False):
                if cluster_of.get(str(row.doc_id)) == seed:
                    seed_ranks.append(int(row.rank))
                    systems_for[seed].add(system)
        best_rank[seed] = min(seed_ranks) if seed_ranks else 10**9
        if not systems_for[seed]:
            systems_for[seed].add("seed")

    def sort_key(cid: str):
        size = int(clusters.loc[cid, "size"]) if cid in clusters.index else 0
        return (0 if cid == seed else 1, best_rank[cid], -size, cid)

    selected = sorted(best_rank, key=sort_key)[:cap]

    rows = []
    for cid in selected:
        meta = clusters.loc[cid] if cid in clusters.index else None
        title = clean(meta["rep_title"]) if meta is not None else ""
        size = int(meta["size"]) if meta is not None else len(cluster_docs.get(cid, []))
        hosts = int(meta["hosts"]) if meta is not None else 0
        earliest = None
        if meta is not None:
            earliest = meta["earliest_publish"]
            if pd.isna(earliest):
                earliest = meta["earliest_warc"]

        candidates = sorted(cluster_docs.get(cid, []), key=lambda x: (x[0], x[1]))
        chosen: list[tuple[str, str]] = []
        for _, doc_id, _ in candidates:
            if doc_id not in doc_meta.index:
                continue
            host = str(doc_meta.loc[doc_id, "host"])
            if all(host != h for h, _ in chosen):
                chosen.append((host, doc_id))
            if len(chosen) == 2:
                break
        if not chosen and meta is not None:
            rep = str(meta["rep_doc_id"])
            if rep in doc_meta.index:
                chosen = [(str(doc_meta.loc[rep, "host"]), rep)]

        snippets = []
        for _, doc_id in chosen:
            snippets.append(clean(str(doc_meta.loc[doc_id, "text"])[:snippet_chars]))
        while len(snippets) < 2:
            snippets.append("")

        rows.append(
            {
                "qid": query.qid,
                "month": query.month,
                "style": getattr(query, "style", ""),
                "query": query.text,
                "cluster_id": cid,
                "best_rank": 999 if best_rank[cid] >= 10**9 else best_rank[cid],
                "systems": "+".join(sorted(systems_for[cid])) or "seed",
                "size": size,
                "hosts": hosts,
                "earliest_date": earliest,
                "title": title,
                "snippet_1": snippets[0],
                "snippet_2": snippets[1],
                "from_seed": cid == seed,
                "seed_forced": seed_forced and cid == seed,
                "grade": "",
                "same_as": "",
                "annotator": "",
                "notes": "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="queries/queries.csv")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--systems", default="bm25,dense")
    parser.add_argument("--per-system", type=int, default=10)
    parser.add_argument("--cap", type=int, default=15)
    parser.add_argument("--snippet-chars", type=int, default=300)
    parser.add_argument("--status", default=None)
    parser.add_argument("--qids", default=None, help="comma-separated qids")
    parser.add_argument("--out", default="queries/candidates.csv")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    docs_dir = resolve(args.docs_dir or cfg["paths"]["docs_dir"])
    runs_dir = resolve(args.runs_dir or cfg["paths"]["runs_dir"])
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    queries = pd.read_csv(resolve(args.queries), dtype={"month": str, "seed_cluster": str})
    if args.status and "status" in queries.columns:
        queries = queries[queries["status"].isin({s.strip() for s in args.status.split(",")})]
    if args.qids:
        wanted = {s.strip() for s in args.qids.split(",")}
        queries = queries[queries["qid"].isin(wanted)]
    queries = queries.dropna(subset=["qid", "text", "month"])

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    runs = {
        system: pd.read_parquet(runs_dir / f"{system}__none.parquet")
        for system in systems
    }

    month_assets: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    rows: list[dict] = []
    for query in queries.itertuples(index=False):
        if query.month not in month_assets:
            month_assets[query.month] = load_month_assets(docs_dir, query.month)
        docs, clusters = month_assets[query.month]
        qrows = build_for_query(
            query,
            runs,
            docs,
            clusters,
            per_system=args.per_system,
            cap=args.cap,
            snippet_chars=args.snippet_chars,
        )
        rows.extend(qrows)
        forced = sum(1 for r in qrows if r["from_seed"] and r["systems"] == "seed")
        print(f"[{query.qid}] {len(qrows)} candidates (seed forced: {forced})")

    candidates = pd.DataFrame(rows)
    candidates.to_csv(out_path, index=False)
    print(f"\n[summary] {len(candidates)} candidates for {queries.qid.nunique()} queries -> {out_path}")
    print(candidates.groupby("qid").size().to_string())


if __name__ == "__main__":
    main()
