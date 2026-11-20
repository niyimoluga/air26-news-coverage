"""S4: hybrid retrieval by Reciprocal Rank Fusion (RRF) of BM25 and dense.

RRF combines ranked lists without score calibration:

    score(d) = sum over systems 1 / (k + rank_system(d))

The fusion is deterministic and needs no training. It is a fusion baseline, not
a reranker, and like every other system it never sees cluster ids or qrels.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def rrf_fuse(
    ranked_lists: Mapping[str, Sequence[str]],
    k: int = 60,
    top_k: int = 100,
) -> pd.DataFrame:
    """Fuse ranked doc-id lists into one ranking.

    ``ranked_lists`` maps a system name to its doc ids in rank order. A doc that
    appears in several lists gets the sum of its reciprocal-rank contributions.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for _system, doc_ids in ranked_lists.items():
        for rank, doc_id in enumerate(doc_ids, start=1):
            doc_id = str(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1

    ranked = sorted(
        scores.items(), key=lambda item: (-item[1], first_seen[item[0]])
    )[:top_k]
    return pd.DataFrame(
        {"doc_id": [d for d, _ in ranked], "score": [s for _, s in ranked]}
    )
