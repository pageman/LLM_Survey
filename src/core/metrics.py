"""Shared metrics for local experiments."""

from __future__ import annotations

import numpy as np


def cross_entropy_from_probs(probabilities: np.ndarray, target_index: int, eps: float = 1e-12) -> float:
    return float(-np.log(float(probabilities[target_index]) + eps))


def perplexity_from_losses(losses: list[float] | np.ndarray) -> float:
    losses = np.asarray(losses, dtype=float)
    return float(np.exp(np.mean(losses)))


def top_k_accuracy(probabilities: np.ndarray, target_index: int, k: int = 1) -> float:
    top_k = np.argsort(probabilities)[::-1][:k]
    return 1.0 if int(target_index) in top_k else 0.0


def compute_retrieval_metrics(
    predictions: list[list[int] | np.ndarray],
    correct_indices: list[int],
    k_values: list[int] | None = None,
) -> tuple[dict[int, float], float]:
    k_values = k_values or [1, 3, 5]
    n_queries = len(predictions)
    recalls = {k: 0 for k in k_values}
    reciprocal_ranks = []

    for pred, correct_idx in zip(predictions, correct_indices):
        pred_list = list(pred)
        if correct_idx in pred_list:
            rank = pred_list.index(correct_idx) + 1
            reciprocal_ranks.append(1.0 / rank)
            for k in k_values:
                if rank <= k:
                    recalls[k] += 1
        else:
            reciprocal_ranks.append(0.0)

    recalls = {k: value / max(n_queries, 1) for k, value in recalls.items()}
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    return recalls, mrr
