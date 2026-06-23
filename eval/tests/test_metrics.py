from __future__ import annotations

import unittest

from eval.metrics import dcg_at_k, hit_rate, mrr, ndcg_at_k


class MetricsTest(unittest.TestCase):
    def test_hit_rate(self) -> None:
        self.assertEqual(hit_rate([10, 20, 30], {20}, 1), 0.0)
        self.assertEqual(hit_rate([10, 20, 30], {20}, 2), 1.0)

    def test_mrr(self) -> None:
        self.assertAlmostEqual(mrr([10, 20, 30], {20}), 0.5)
        self.assertEqual(mrr([10, 20, 30], {99}), 0.0)

    def test_dcg(self) -> None:
        score = dcg_at_k([10, 20, 30], {10, 30}, 3)
        self.assertAlmostEqual(score, 1.0 + 1.0 / 2.0)

    def test_ndcg(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([10, 20, 30], {10, 30}, 3), 0.9197207891481876)
        self.assertEqual(ndcg_at_k([10, 20, 30], set(), 3), 0.0)


if __name__ == "__main__":
    unittest.main()
