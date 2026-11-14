"""P5: Cohen's kappa for the double-judged pilot subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from ..data.common import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", default="queries/judgments.csv")
    parser.add_argument("--double", default="queries/judgments_double.csv")
    parser.add_argument("--out", default="results/metrics/pilot_kappa.csv")
    args = parser.parse_args()

    first = pd.read_csv(resolve(args.judgments), dtype={"qid": str, "cluster_id": str})
    second = pd.read_csv(resolve(args.double), dtype={"qid": str, "cluster_id": str})
    first = first[["qid", "cluster_id", "grade"]]
    second = second[["qid", "cluster_id", "grade"]]
    merged = first.merge(second, on=["qid", "cluster_id"], suffixes=("_1", "_2"))
    merged["grade_1"] = pd.to_numeric(merged["grade_1"], errors="coerce")
    merged["grade_2"] = pd.to_numeric(merged["grade_2"], errors="coerce")
    merged = merged.dropna(subset=["grade_1", "grade_2"])

    if len(merged) < 5:
        raise SystemExit(f"only {len(merged)} double-judged pairs; need at least 5")
    if merged[["grade_1", "grade_2"]].isin([0, 1, 2]).all().all() is False:
        raise SystemExit("grades must be in {0,1,2}")

    a = merged["grade_1"].astype(int).to_numpy()
    b = merged["grade_2"].astype(int).to_numpy()
    kappa = cohen_kappa_score(a, b)
    kappa_qw = cohen_kappa_score(a, b, weights="quadratic")
    agreement = float((a == b).mean())
    cm = confusion_matrix(a, b, labels=[0, 1, 2])

    out_path = Path(resolve(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "n": len(merged),
                "kappa": kappa,
                "kappa_quadratic": kappa_qw,
                "agreement": agreement,
            }
        ]
    ).to_csv(out_path, index=False)

    print(f"double-judged pairs: {len(merged)}")
    print(f"agreement:           {agreement:.1%}")
    print(f"Cohen's kappa:       {kappa:.3f}")
    print(f"quadratic weighted:  {kappa_qw:.3f}")
    print("confusion matrix (rows=annotator 1, cols=annotator 2):")
    print(pd.DataFrame(cm, index=[0, 1, 2], columns=[0, 1, 2]).to_string())
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
