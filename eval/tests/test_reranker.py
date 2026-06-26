import unittest

from eval.reranker import rerank_hits
from video_rag_cli import SearchHit


class StubReranker:
    def __init__(self, scores):
        self.scores = scores

    def score_pairs(self, pairs):
        return self.scores[: len(pairs)]


class RerankerTest(unittest.TestCase):
    def test_rerank_hits_reorders_by_rerank_score(self):
        hits = [
            SearchHit(1, 1, "a.mp4", 0.0, 5.0, "visual", "multi_frame", "first", 0.9, "hybrid"),
            SearchHit(2, 1, "a.mp4", 5.0, 10.0, "visual", "multi_frame", "second", 0.8, "hybrid"),
        ]
        reranked, debug_rows = rerank_hits("query", hits, StubReranker([0.1, 0.9]), top_k=2)
        self.assertEqual([hit.segment_id for hit in reranked], [2, 1])
        self.assertEqual(debug_rows[0]["segment_id"], 2)
        self.assertEqual(debug_rows[0]["after_rank"], 1)

    def test_rerank_hits_skips_empty_content(self):
        hits = [
            SearchHit(1, 1, "a.mp4", 0.0, 5.0, "visual", "multi_frame", "", 0.9, "hybrid"),
            SearchHit(2, 1, "a.mp4", 5.0, 10.0, "visual", "multi_frame", "second", 0.8, "hybrid"),
        ]
        reranked, debug_rows = rerank_hits("query", hits, StubReranker([0.7]), top_k=2)
        self.assertEqual([hit.segment_id for hit in reranked], [2])
        empty_row = next(row for row in debug_rows if row["segment_id"] == 1)
        self.assertTrue(empty_row["empty_content"])
        self.assertIsNone(empty_row["after_rank"])
