"""P1: filter one parquet shard per month into docs_YYYY-MM.parquet.

Input:  data/manifest.csv + cached Infini-News shards
Output: data/processed/docs_YYYY-MM.parquet
        data/processed/corpus_stats.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .common import (
    ensure_shard,
    load_config,
    load_manifest,
    resolve,
    select_months,
)

COLUMNS = [
    "url",
    "url_hostname",
    "warc_date",
    "publish_date",
    "title",
    "text",
    "text_length",
    "text_xxhash64",
    "warc_record_id",
    "http_status",
    "language_iso639_3",
]

OUT_COLUMNS = [
    "doc_id",
    "month",
    "host",
    "title",
    "text",
    "publish_date",
    "warc_date",
    "text_length",
    "text_hash",
    "url",
]


def process_month(
    row: dict,
    *,
    cache_dir: str,
    out_dir: Path,
    language: str,
    min_chars: int,
    max_chars: int,
    truncate_chars: int,
    download: bool,
    overwrite: bool,
) -> dict:
    month = row["month"]
    out_path = out_dir / f"docs_{month}.parquet"
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path.name} exists")
        return {"month": month, "skipped": True}

    started = time.time()
    shard = ensure_shard(cache_dir, row["shard"], download=download)
    table = pq.read_table(shard, columns=COLUMNS)
    df = table.to_pandas()

    keep = (
        df["language_iso639_3"].eq(language)
        & df["text_length"].fillna(0).between(min_chars, max_chars)
        & df["http_status"].fillna(200).eq(200)
        & df["warc_record_id"].notna()
        & df["text"].notna()
    )
    df = df.loc[keep].copy()

    # stable document id; suffix the rare duplicate warc_record_id
    doc_ids = df["warc_record_id"].astype(str)
    duplicated = doc_ids.duplicated(keep="first")
    if duplicated.any():
        occurrence = doc_ids.groupby(doc_ids).cumcount().astype(str)
        doc_ids = doc_ids.where(~duplicated, doc_ids + "#" + occurrence)
        print(f"[warn] {month}: {int(duplicated.sum())} duplicate doc ids suffixed")

    out = pd.DataFrame(
        {
            "doc_id": doc_ids.to_numpy(),
            "month": month,
            "host": df["url_hostname"].fillna("").astype(str).to_numpy(),
            "title": df["title"].fillna("").astype(str).to_numpy(),
            "text": df["text"].astype(str).str.slice(0, truncate_chars).to_numpy(),
            "publish_date": df["publish_date"].astype("string").to_numpy(),
            "warc_date": pd.to_datetime(
                df["warc_date"], utc=True, errors="coerce"
            ).to_numpy(),
            "text_length": df["text_length"].fillna(0).astype("int64").to_numpy(),
            "text_hash": df["text_xxhash64"].astype(str).to_numpy(),
            "url": df["url"].fillna("").astype(str).to_numpy(),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(out_path)

    stats = {
        "month": month,
        "source_shard": row["shard"],
        "rows_raw": int(row["rows_raw"]),
        "rows_manifest_filtered": int(row["rows_filtered"]),
        "rows_written": int(len(out)),
        "hosts": int(out["host"].nunique()),
        "median_text_length": float(out["text_length"].median()) if len(out) else 0.0,
        "total_chars": int(out["text_length"].sum()),
        "runtime_s": round(time.time() - started, 1),
    }
    print(
        f"[done] {month}: {stats['rows_written']:,} docs, "
        f"{stats['hosts']:,} hosts, {stats['runtime_s']}s"
    )
    if stats["rows_written"] != stats["rows_manifest_filtered"]:
        print(
            f"[warn] {month}: written {stats['rows_written']:,} vs manifest "
            f"{stats['rows_manifest_filtered']:,}"
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--cache-dir", default=None, help="override config cache_dir")
    parser.add_argument("--out-dir", default=None, help="override config docs_dir")
    parser.add_argument("--months", default=None, help="e.g. 2023-01,2023-06")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download", action="store_true", help="fetch missing shards")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre = cfg["preprocess"]
    cache_dir = args.cache_dir or cfg["paths"]["cache_dir"]
    out_dir = resolve(args.out_dir or cfg["paths"]["docs_dir"])

    manifest = select_months(load_manifest(args.manifest), args.months)
    stats = []
    for row in manifest.to_dict("records"):
        stats.append(
            process_month(
                row,
                cache_dir=cache_dir,
                out_dir=out_dir,
                language=pre["language"],
                min_chars=int(pre["min_chars"]),
                max_chars=int(pre["max_chars"]),
                truncate_chars=int(pre["truncate_chars"]),
                download=args.download,
                overwrite=args.overwrite,
            )
        )

    stats_df = pd.DataFrame(stats)
    stats_path = out_dir / "corpus_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    if "skipped" in stats_df.columns:
        done = stats_df[~stats_df["skipped"].fillna(False).astype(bool)]
    else:
        done = stats_df
    if len(done) and "rows_written" in done.columns:
        print(
            f"\n[summary] {len(done)} month(s), "
            f"{int(done['rows_written'].sum()):,} docs, "
            f"{int(done['hosts'].sum()):,} host-month rows"
        )
        if int(done["rows_written"].sum()) == 0:
            sys.exit("no documents written; check filters")
    print(f"[summary] stats -> {stats_path}")


if __name__ == "__main__":
    main()
