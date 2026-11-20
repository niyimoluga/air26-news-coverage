"""S3: cross-encoder reranking of a candidate list.

The cross-encoder reads the query and one document together and outputs a
relevance score. It is applied to the top-k candidates of a base system
(BM25 by default), so the cost is one forward pass per candidate, not per
document in the month.

This module never sees cluster ids or qrels: it only receives document texts.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Truncate the document side; the MS MARCO cross-encoders were trained on
# passages, and long news articles would otherwise blow up the token budget.
MAX_CHARS = 1200


def load_model(model_name: str = DEFAULT_MODEL, device: str | None = None):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


class CrossEncoderRetriever:
    """Rerank candidates ``(doc_id, text)`` for one query."""

    def __init__(self, model, max_chars: int = MAX_CHARS, batch_size: int = 32):
        self.model = model
        self.max_chars = max_chars
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        k: int = 100,
    ) -> pd.DataFrame:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        if not doc_ids:
            return pd.DataFrame({"doc_id": [], "score": []})

        pairs = [
            (query, (text or "")[: self.max_chars]) for text in texts
        ]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        scores = np.asarray(scores, dtype=float).reshape(-1)
        order = np.argsort(-scores)[:k]
        return pd.DataFrame(
            {
                "doc_id": [doc_ids[i] for i in order],
                "score": [float(scores[i]) for i in order],
            }
        )
