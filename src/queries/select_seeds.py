"""P3 helper: list seed story clusters for query authoring.

Input:  data/processed/clusters_YYYY-MM.parquet + docs_YYYY-MM.parquet
Output: queries/seed_candidates.csv (gitignored; contains titles/snippets)

Filters out template/listicle clusters (price pages, live blogs, etc.) and
clusters that are one host repeating itself. One row per candidate cluster,
with the representative article's URL and a short snippet so an annotator can
write a query without opening the corpus.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from ..data.common import load_config, resolve, select_months, load_manifest

DEFAULT_BLOCKLIST = (
    r"on road price|price in|photo of the day|live updates?|live blog|"
    r"horoscope|lottery|weather forecast|match preview|how to watch|"
    r"streaming guide|results today|transfer rumou?rs|quiz|crossword|"
    r"obituary|classifieds|property for sale|real estate|recruitment|"
    r"sponsored|deal of the day|coupon|discount code|"
    r"prnewswire|business wire|globenewswire|accesswire|einpresswire|press release"
)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-dir", default=None)
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="queries/seed_candidates.csv")
    parser.add_argument("--months", default=None)
    parser.add_argument("--min-size", type=int, default=5)
    parser.add_argument("--min-hosts", type=int, default=3)
    parser.add_argument("--max-ratio", type=float, default=10.0,
                        help="drop clusters with size/hosts above this (one host repeating)")
    parser.add_argument("--per-month", type=int, default=12)
    parser.add_argument("--blocklist", default=DEFAULT_BLOCKLIST)
    parser.add_argument(
        "--exclude-url",
        default=r"prnewswire|businesswire|globenewswire|accesswire|einpresswire",
        help="drop press-release domains (they syndicate heavily and are not news events)",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    clusters_dir = resolve(args.clusters_dir or cfg["paths"]["clusters_dir"])
    docs_dir = resolve(args.docs_dir or cfg["paths"]["docs_dir"])
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = select_months(load_manifest(args.manifest), args.months)
    pattern = re.compile(args.blocklist, re.IGNORECASE)

    rows = []
    for month in manifest["month"]:
        cpath = clusters_dir / f"clusters_{month}.parquet"
        dpath = docs_dir / f"docs_{month}.parquet"
        if not cpath.exists() or not dpath.exists():
            print(f"[warn] missing {month}; run P1/P2 first")
            continue
        clusters = pd.read_parquet(cpath)
        docs = pd.read_parquet(dpath, columns=["doc_id", "url", "text"])

        clusters = clusters[
            (clusters["size"] >= args.min_size)
            & (clusters["hosts"] >= args.min_hosts)
        ].copy()
        clusters["ratio"] = clusters["size"] / clusters["hosts"].clip(lower=1)
        clusters = clusters[clusters["ratio"] <= args.max_ratio]
        clusters["rep_title"] = clusters["rep_title"].fillna("").map(clean)
        blocked = clusters["rep_title"].str.contains(pattern, na=False)
        print(f"[{month}] eligible={len(clusters):,} blocked={int(blocked.sum()):,}")
        clusters = clusters[~blocked]

        join = docs.set_index("doc_id")[["url", "text"]]
        clusters = clusters.join(join, on="rep_doc_id")
        if args.exclude_url:
            url_mask = clusters["url"].fillna("").str.contains(
                args.exclude_url, case=False, regex=True
            )
            print(f"[{month}] press-release domains dropped={int(url_mask.sum()):,}")
            clusters = clusters[~url_mask]
        clusters["seed_snippet"] = (
            clusters["text"].fillna("").str.slice(0, 400).map(clean)
        )
        snippet_blocked = (
            clusters["rep_title"].fillna("") + " " + clusters["seed_snippet"].fillna("")
        ).str.contains(pattern, na=False)
        if snippet_blocked.any():
            print(f"[{month}] title/snippet blocklist dropped={int(snippet_blocked.sum()):,}")
            clusters = clusters[~snippet_blocked]
        clusters["month"] = month
        clusters = clusters.sort_values(
            ["size", "hosts"], ascending=False
        ).head(args.per_month)
        clusters["rank"] = range(1, len(clusters) + 1)
        rows.append(
            clusters[
                [
                    "month",
                    "rank",
                    "cluster_id",
                    "size",
                    "hosts",
                    "ratio",
                    "earliest_publish",
                    "earliest_warc",
                    "rep_title",
                    "rep_doc_id",
                    "url",
                    "seed_snippet",
                ]
            ]
        )

    if not rows:
        raise SystemExit("no candidates; check paths and filters")

    result = pd.concat(rows, ignore_index=True)
    result.to_csv(out_path, index=False)
    print(f"\n[summary] {len(result):,} candidates -> {out_path}")
    print(result.groupby("month").size().to_string())


if __name__ == "__main__":
    main()
