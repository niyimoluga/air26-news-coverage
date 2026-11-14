"""Validate pilot judgments against the candidate pool.

Checks: every judgment maps to a candidate, no duplicate (qid, cluster_id),
grades in {0,1,2}, same_as points to a candidate of the same query, and
(optionally) every candidate is judged.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ..data.common import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="queries/candidates.csv")
    parser.add_argument("--judgments", default="queries/judgments.csv")
    parser.add_argument("--require-complete", action="store_true",
                        help="fail if any candidate has no judgment")
    parser.add_argument("--allow-empty", action="store_true",
                        help="only structural checks (for the blank template)")
    args = parser.parse_args()

    candidates = pd.read_csv(resolve(args.candidates))
    judgments = pd.read_csv(resolve(args.judgments))
    errors: list[str] = []

    required = {"qid", "cluster_id", "grade"}
    if required - set(judgments.columns):
        sys.exit(f"judgments missing columns: {sorted(required - set(judgments.columns))}")

    key_c = set(zip(candidates["qid"].astype(str), candidates["cluster_id"].astype(str)))
    key_j = list(zip(judgments["qid"].astype(str), judgments["cluster_id"].astype(str)))

    if len(key_j) != len(set(key_j)):
        errors.append("duplicate (qid, cluster_id) rows")
    unknown = set(key_j) - key_c
    if unknown:
        errors.append(f"{len(unknown)} judgments not in candidates, e.g. {list(unknown)[:3]}")

    if not args.allow_empty:
        grades = pd.to_numeric(judgments["grade"], errors="coerce")
        if grades.isna().any():
            errors.append(f"{int(grades.isna().sum())} missing/non-numeric grades")
        bad = set(grades.dropna().unique()) - {0, 1, 2}
        if bad:
            errors.append(f"grades outside {{0,1,2}}: {sorted(bad)}")
        if args.require_complete:
            missing = key_c - set(key_j)
            if missing:
                errors.append(f"{len(missing)} candidates unjudged, e.g. {list(missing)[:3]}")

        if "same_as" in judgments.columns:
            same = judgments["same_as"].astype("object").fillna("").astype(str).str.strip()
            for row in judgments[same != ""].itertuples(index=False):
                if str(row.same_as).strip() == str(row.cluster_id):
                    errors.append(f"{row.qid}/{row.cluster_id}: same_as points to itself")
                if (str(row.qid), str(row.same_as).strip()) not in key_c:
                    errors.append(f"{row.qid}/{row.cluster_id}: same_as {row.same_as} not in candidates")

    if errors:
        print("INVALID")
        for error in errors[:20]:
            print(" -", error)
        sys.exit(1)
    print(
        f"OK: {len(judgments)} judgments, {judgments.qid.nunique()} queries, "
        f"{len(key_c)} candidates"
    )


if __name__ == "__main__":
    main()
