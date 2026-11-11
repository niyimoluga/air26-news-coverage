"""Shared helpers for the AIR26 news-coverage project."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

HF_RESOLVE = (
    "https://huggingface.co/datasets/ruggsea/infini-news-corpus/resolve/main/"
)


def repo_root() -> Path:
    """Return the repository root (this file is src/data/common.py)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Load config/default.yaml, or an explicit YAML file."""
    cfg_path = Path(path) if path is not None else repo_root() / "config/default.yaml"
    with cfg_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else (repo_root() / path)


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read the 24-shard sample manifest (one row per month)."""
    df = pd.read_csv(resolve(path), dtype={"month": str})
    expected = {"month", "shard", "rows_raw", "rows_filtered"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return df


def select_months(manifest: pd.DataFrame, months: str | None) -> pd.DataFrame:
    """Optionally restrict the manifest to a comma-separated list of months."""
    if not months:
        return manifest.reset_index(drop=True)
    wanted = {m.strip() for m in months.split(",") if m.strip()}
    selected = manifest[manifest["month"].isin(wanted)].reset_index(drop=True)
    missing = wanted - set(selected["month"])
    if missing:
        raise ValueError(f"months not in manifest: {sorted(missing)}")
    return selected


def shard_url(shard_path: str) -> str:
    """Hugging Face URL for one parquet shard path from the manifest."""
    return HF_RESOLVE + shard_path.lstrip("/")


def cache_file(cache_dir: str | Path, shard_path: str) -> Path:
    """Local cache path for a shard (basename only)."""
    return resolve(cache_dir) / Path(shard_path).name


def ensure_shard(
    cache_dir: str | Path, shard_path: str, download: bool = False
) -> Path:
    """Return the cached shard, optionally downloading it from Hugging Face."""
    target = cache_file(cache_dir, shard_path)
    if target.exists() and target.stat().st_size > 0:
        return target
    if not download:
        raise FileNotFoundError(
            f"missing shard {target}; pass --download or copy it into the cache"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(shard_url(shard_path), timeout=240) as response:
        tmp.write_bytes(response.read())
    tmp.rename(target)
    return target


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (English news)."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_shingles(text: str, k: int = 5) -> list[str]:
    """Word k-gram shingles; short texts become one shingle."""
    words = normalise(text).split()
    if not words:
        return []
    if len(words) < k:
        return [" ".join(words)]
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def iter_months(manifest: pd.DataFrame) -> Iterable[str]:
    return manifest["month"].tolist()
