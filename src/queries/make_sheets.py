"""P5: create the pilot annotation sheets and collect the judgments.

Modes
-----
init     : read candidates.csv, pick ~20 % of each query for double judging,
           write candidates.csv (first annotator), double_candidates.csv
           (second annotator) and blank judgments.csv / judgments_double.csv.
collect  : read the filled candidates.csv and double_candidates.csv, validate,
           and write slim judgments.csv / judgments_double.csv (no article text).

candidates.csv and double_candidates.csv contain titles/snippets and are
gitignored; judgments.csv and judgments_double.csv contain only IDs and grades
and are safe to commit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..data.common import resolve

SLIM = ["qid", "cluster_id", "grade", "same_as", "annotator", "notes"]


def pick_double(candidates: pd.DataFrame, per_query: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, group in candidates.groupby("qid", sort=True):
        n = min(per_query, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def init_sheets(candidates_path: Path, out_dir: Path, per_query: int, seed: int,
                force: bool) -> None:
    candidates = pd.read_csv(candidates_path)
    for col in SLIM:
        if col not in candidates.columns:
            candidates[col] = ""
        candidates[col] = candidates[col].astype("object").fillna("")

    double_path = out_dir / "double_candidates.csv"
    if double_path.exists() and not force:
        raise SystemExit(f"{double_path} exists; pass --force to overwrite")
    double = pick_double(candidates, per_query, seed)
    for col in ["grade", "same_as", "annotator", "notes"]:
        double[col] = ""
    double.to_csv(double_path, index=False)

    candidates.to_csv(candidates_path, index=False)
    candidates[SLIM].to_csv(out_dir / "judgments.csv", index=False)
    double[SLIM].to_csv(out_dir / "judgments_double.csv", index=False)

    print(f"[init] candidates: {len(candidates)} rows ({candidates.qid.nunique()} queries)")
    print(f"[init] double:     {len(double)} rows -> {double_path.name}")
    print(f"[init] sheets:     judgments.csv, judgments_double.csv")
    print("[next] fill candidates.csv (annotator 1) and double_candidates.csv (annotator 2),")
    print("       then run: python -m src.queries.make_sheets --mode collect")


def collect(candidates_path: Path, out_dir: Path) -> None:
    candidates = pd.read_csv(candidates_path)
    double_path = out_dir / "double_candidates.csv"
    if not double_path.exists():
        raise SystemExit(f"missing {double_path}")
    double = pd.read_csv(double_path)

    for name, frame in (("candidates", candidates), ("double", double)):
        if frame["grade"].isna().any() or (frame["grade"].astype(str).str.strip() == "").any():
            raise SystemExit(f"{name}: empty grades remain")
        frame["grade"] = pd.to_numeric(frame["grade"], errors="coerce")

    candidates[SLIM].to_csv(out_dir / "judgments.csv", index=False)
    double[SLIM].to_csv(out_dir / "judgments_double.csv", index=False)
    print(f"[collect] judgments.csv: {len(candidates)} rows")
    print(f"[collect] judgments_double.csv: {len(double)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["init", "collect"], default="init")
    parser.add_argument("--candidates", default="queries/candidates.csv")
    parser.add_argument("--out-dir", default="queries")
    parser.add_argument("--double-per-query", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    candidates_path = resolve(args.candidates)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "init":
        init_sheets(candidates_path, out_dir, args.double_per_query, args.seed, args.force)
    else:
        collect(candidates_path, out_dir)


if __name__ == "__main__":
    main()
