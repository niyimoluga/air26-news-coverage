"""P3/P6 helper: list seed story clusters for query authoring.

Input:  data/processed/clusters_YYYY-MM.parquet + docs_YYYY-MM.parquet
Output: queries/seed_candidates.csv (gitignored; contains titles/snippets)

Filters out template/listicle clusters (price pages, live blogs, etc.) and
clusters that are one host repeating itself. One row per candidate cluster,
with the representative article's URL and a short snippet so an annotator can
write a query without opening the corpus.

Richness ranking
----------------
The pilot showed that raw cluster size is a poor seed criterion: a one-report
event can look like 13 clusters because wire copies fragment. ``--richness``
therefore counts, for each seed, how many *distinct headlines* in the same
month cover the same event (title tokens shared with the seed, rare tokens
only). That is a cheap proxy for "how many different stories exist", which is
what the evaluation needs. Prefer seeds with richness >= 3.
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
    r"\bWFO\b|warnings?, watches|zone forecast|marine warning|coastal waters forecast|"
    r"forecast discussion|hazardous weather outlook"
    r"|player props|current conditions|best bets|prop bets|parlay"
)

STOPWORDS = set(
    """the a an and or of to in on for with at by from as is are was were be been
    this that these those its it his her their our your after before over under
    new says said say amid into about more than not will would can could may
    also but if then so what who whom which when where why how during since
    while against between among across off out up down one two three""".split()
)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def norm_title(title: str) -> str:
    """Lowercase, strip outlet suffix, drop stopwords and short tokens."""
    text = clean(title)
    # "Headline - Outlet" / "Headline | Outlet" / "Headline – Outlet"
    text = re.sub(r"\s+[|\-–—]\s+[^|\-–—]{2,40}$", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def title_tokens(title: str) -> set[str]:
    return set(norm_title(title).split())


def richness_scores(
    clusters: pd.DataFrame,
    min_shared: int = 2,
    min_jaccard: float = 0.3,
) -> pd.DataFrame:
    """Add ``n_headline_groups`` and event-size columns to a month's clusters.

    Two clusters are treated as covering the same event when their normalised
    titles share at least ``min_shared`` content tokens and reach a token
    Jaccard of ``min_jaccard`` (or share three or more tokens). Among those
    neighbours we count distinct normalised headlines, hosts and documents.

    This is a title-level proxy, not a judgement: it finds events with several
    *different* headlines, which is what the pilot showed raw cluster size
    cannot tell apart.
    """
    clusters = clusters.copy()
    clusters["_tokens"] = clusters["rep_title"].fillna("").map(title_tokens)
    clusters["_norm"] = clusters["rep_title"].fillna("").map(norm_title)

    token_sets = clusters["_tokens"].tolist()
    norms = clusters["_norm"].tolist()
    hosts = clusters["hosts"].fillna(1).astype(int).tolist()
    sizes = clusters["size"].fillna(1).astype(int).tolist()

    groups: list[int] = []
    event_hosts: list[int] = []
    event_docs: list[int] = []
    for i, tokens in enumerate(token_sets):
        if not tokens:
            groups.append(1)
            event_hosts.append(int(hosts[i]))
            event_docs.append(int(sizes[i]))
            continue
        headlines: set[str] = set()
        hosts_total = 0
        docs_total = 0
        for j, other in enumerate(token_sets):
            shared = len(tokens & other)
            if shared < min_shared:
                continue
            union = len(tokens | other)
            jaccard = shared / union if union else 0.0
            if shared >= 3 or jaccard >= min_jaccard:
                headlines.add(norms[j])
                hosts_total += int(hosts[j])
                docs_total += int(sizes[j])
        groups.append(max(1, len(headlines)))
        event_hosts.append(max(int(hosts[i]), hosts_total))
        event_docs.append(max(int(sizes[i]), docs_total))

    clusters["n_headline_groups"] = groups
    clusters["event_hosts"] = event_hosts
    clusters["event_docs"] = event_docs
    return clusters


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
    parser.add_argument("--richness", action="store_true",
                        help="rank by distinct headlines per event instead of raw size")
    parser.add_argument("--min-richness", type=int, default=1,
                        help="drop seeds with fewer distinct headlines than this")
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

        if clusters.empty:
            print(f"[{month}] nothing left after filters")
            continue

        if args.richness:
            clusters = richness_scores(clusters)
            clusters = clusters[clusters["n_headline_groups"] >= args.min_richness]
            clusters = clusters.sort_values(
                ["n_headline_groups", "size", "hosts"], ascending=False
            )
        else:
            clusters = clusters.sort_values(["size", "hosts"], ascending=False)

        clusters["month"] = month
        clusters = clusters.head(args.per_month)
        clusters["rank"] = range(1, len(clusters) + 1)
        columns = [
            "month", "rank", "cluster_id", "size", "hosts", "ratio",
            "earliest_publish", "earliest_warc", "rep_title", "rep_doc_id",
            "url", "seed_snippet",
        ]
        if args.richness:
            columns += ["n_headline_groups", "event_hosts", "event_docs"]
        rows.append(clusters[columns])

    if not rows:
        raise SystemExit("no candidates; check paths and filters")

    result = pd.concat(rows, ignore_index=True)
    result.to_csv(out_path, index=False)
    print(f"\n[summary] {len(result):,} candidates -> {out_path}")
    print(result.groupby("month").size().to_string())
    if args.richness:
        print("\n[richness] distribution:")
        print(result["n_headline_groups"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
