"""Ranking metrics for retrieval evaluation.

Pure functions over (ranked_ids, relevance_map) so they can be unit-tested
against hand-computed values without touching the index or an LLM.

Relevance is graded:
    2 = strong match (would shortlist)
    1 = partial match (plausible, weaker on a key requirement)
    0 = irrelevant
Binary metrics (recall/precision/MRR) treat grade >= 1 as relevant.
"""

import math


def _relevant_ids(relevance: dict, threshold: int = 1) -> set:
    return {cid for cid, grade in relevance.items() if grade >= threshold}


def recall_at_k(ranked_ids: list, relevance: dict, k: int, threshold: int = 1) -> float:
    """Share of all relevant candidates that appear in the top k."""
    relevant = _relevant_ids(relevance, threshold)
    if not relevant:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hits / len(relevant)


def precision_at_k(ranked_ids: list, relevance: dict, k: int, threshold: int = 1) -> float:
    """Share of the top k that is relevant."""
    if k <= 0:
        return 0.0
    relevant = _relevant_ids(relevance, threshold)
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hits / k


def reciprocal_rank(ranked_ids: list, relevance: dict, threshold: int = 1) -> float:
    """1 / rank of the first relevant result, else 0."""
    relevant = _relevant_ids(relevance, threshold)
    for position, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(ranked_ids: list, relevance: dict, k: int) -> float:
    """Discounted cumulative gain using the (2^rel - 1) gain formulation."""
    total = 0.0
    for position, cid in enumerate(ranked_ids[:k], start=1):
        grade = relevance.get(cid, 0)
        if grade > 0:
            total += (2**grade - 1) / math.log2(position + 1)
    return total


def ndcg_at_k(ranked_ids: list, relevance: dict, k: int) -> float:
    """DCG normalised by the best achievable ordering (1.0 = perfect ranking)."""
    ideal_order = sorted(relevance, key=lambda cid: relevance[cid], reverse=True)
    ideal = dcg_at_k(ideal_order, relevance, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevance, k) / ideal


def evaluate_query(ranked_ids: list, relevance: dict, k: int = 5) -> dict:
    """All metrics for one query's ranking."""
    return {
        f"recall@{k}": recall_at_k(ranked_ids, relevance, k),
        f"precision@{k}": precision_at_k(ranked_ids, relevance, k),
        f"ndcg@{k}": ndcg_at_k(ranked_ids, relevance, k),
        "mrr": reciprocal_rank(ranked_ids, relevance),
    }


def aggregate(per_query: list) -> dict:
    """Mean of each metric across queries (macro average)."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: sum(q[key] for q in per_query) / len(per_query) for key in keys}
