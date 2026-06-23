from __future__ import annotations

import math
from collections.abc import Iterable


def hit_rate(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_ids = retrieved_ids[:k]
    return 1.0 if any(item_id in relevant_ids for item_id in top_ids) else 0.0


def mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    score = 0.0
    for rank, item_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if item_id in relevant_ids else 0.0
        if rel:
            score += rel / math.log2(rank + 1)
    return score


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0
    actual = dcg_at_k(retrieved_ids, relevant_ids, k)
    ideal_hits = min(len(relevant_ids), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal == 0:
        return 0.0
    return actual / ideal


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
