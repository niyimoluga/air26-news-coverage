"""S1: BM25 retrieval over one month's pool (bm25s)."""

from __future__ import annotations

from typing import Iterable

import bm25s
import pandas as pd


class BM25Retriever:
    """One BM25 index over a month's documents."""

    def __init__(
        self,
        texts: Iterable[str],
        doc_ids: Iterable[str],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        stopwords: str | None = "en",
    ) -> None:
        self.doc_ids = list(doc_ids)
        self.stopwords = stopwords
        tokens = bm25s.tokenize(
            list(texts), stopwords=stopwords, show_progress=False
        )
        self.retriever = bm25s.BM25(k1=k1, b=b)
        self.retriever.index(tokens, show_progress=False)

    def search(self, query: str, k: int = 100) -> pd.DataFrame:
        k = min(k, len(self.doc_ids))
        query_tokens = bm25s.tokenize(
            [query], stopwords=self.stopwords, show_progress=False
        )
        results, scores = self.retriever.retrieve(
            query_tokens, k=k, show_progress=False
        )
        indices = results[0].tolist()
        values = scores[0].tolist()
        return pd.DataFrame(
            {
                "doc_id": [self.doc_ids[i] for i in indices],
                "score": [float(v) for v in values],
            }
        )
