"""S2: dense bi-encoder retrieval over one month's pool (MiniLM, numpy)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_model(model_name: str = DEFAULT_MODEL, device: str | None = None) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 256
    return model


def encode_documents(
    model: SentenceTransformer,
    texts: Iterable[str],
    *,
    batch_size: int = 64,
) -> np.ndarray:
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


class DenseRetriever:
    """Exact cosine search with a precomputed document matrix."""

    def __init__(
        self,
        model: SentenceTransformer,
        doc_ids: Iterable[str],
        embeddings: np.ndarray,
    ) -> None:
        self.model = model
        self.doc_ids = list(doc_ids)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(self.doc_ids) != len(self.embeddings):
            raise ValueError("doc_ids and embeddings length mismatch")

    def search(self, query: str, k: int = 100) -> pd.DataFrame:
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        scores = self.embeddings @ query_vec
        k = min(k, len(scores))
        if k < len(scores):
            indices = np.argpartition(-scores, k - 1)[:k]
        else:
            indices = np.arange(len(scores))
        indices = indices[np.argsort(-scores[indices])]
        return pd.DataFrame(
            {
                "doc_id": [self.doc_ids[i] for i in indices],
                "score": [float(s) for s in scores[indices]],
            }
        )
