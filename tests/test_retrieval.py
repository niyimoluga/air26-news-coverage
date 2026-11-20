"""Tests for fusion and seed-richness scoring."""

import unittest

import pandas as pd

from src.queries.select_seeds import norm_title, richness_scores
from src.retrieval.hybrid import rrf_fuse


class TestRRF(unittest.TestCase):
    def test_agreement_beats_single_list(self):
        fused = rrf_fuse(
            {"bm25": ["a", "b", "c"], "dense": ["a", "c", "b"]}, k=60, top_k=3
        )
        self.assertEqual(fused["doc_id"].tolist(), ["a", "b", "c"])

    def test_scores_are_reciprocal_ranks(self):
        fused = rrf_fuse({"bm25": ["a"], "dense": ["a"]}, k=60, top_k=5)
        self.assertAlmostEqual(fused.loc[0, "score"], 2 / 61)

    def test_disjoint_lists_keep_first_seen_order(self):
        fused = rrf_fuse({"bm25": ["a"], "dense": ["b"]}, k=60, top_k=5)
        self.assertEqual(fused["doc_id"].tolist(), ["a", "b"])

    def test_top_k_is_respected(self):
        fused = rrf_fuse({"bm25": ["a", "b", "c", "d"]}, k=60, top_k=2)
        self.assertEqual(len(fused), 2)


class TestSeedRichness(unittest.TestCase):
    def test_norm_title_strips_outlet_suffix(self):
        self.assertEqual(norm_title("Storm hits coast - Daily News"), "storm hits coast")

    def test_richness_counts_distinct_headlines(self):
        clusters = pd.DataFrame(
            {
                "rep_title": [
                    "Storm hits coast",
                    "Storm hits coast - Daily News",
                    "Storm batters the coast",
                    "Storm hits coast: three dead",
                    "Unrelated football result",
                ],
                "size": [5, 4, 3, 2, 9],
                "hosts": [3, 3, 2, 2, 4],
            }
        )
        scored = richness_scores(clusters)
        # The four storm variants share "storm" + "coast"/"hits"; the football
        # story shares nothing.
        self.assertGreaterEqual(int(scored.loc[0, "n_headline_groups"]), 2)
        self.assertEqual(int(scored.loc[4, "n_headline_groups"]), 1)


if __name__ == "__main__":
    unittest.main()
