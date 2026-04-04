"""Memorization-versus-generalization demo with bucketed transfer behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MemorizationGeneralizationDemo:
    def evaluate(self) -> dict[str, object]:
        buckets = [
            {"bucket": "exact_recall", "train_score": 0.98, "eval_score": 0.92, "novelty": 0.05},
            {"bucket": "near_copy", "train_score": 0.95, "eval_score": 0.79, "novelty": 0.18},
            {"bucket": "template_transfer", "train_score": 0.89, "eval_score": 0.73, "novelty": 0.44},
            {"bucket": "paraphrase_transfer", "train_score": 0.84, "eval_score": 0.71, "novelty": 0.63},
            {"bucket": "novel_composition", "train_score": 0.78, "eval_score": 0.58, "novelty": 0.91},
        ]
        train = np.array([item["train_score"] for item in buckets], dtype=float)
        eval_scores = np.array([item["eval_score"] for item in buckets], dtype=float)
        novelty = np.array([item["novelty"] for item in buckets], dtype=float)
        gaps = train - eval_scores
        memorization_mask = novelty <= 0.2
        generalization_mask = novelty >= 0.4
        worst_bucket = buckets[int(np.argmax(gaps))]
        return {
            "bucket_scores": [
                {
                    "bucket": item["bucket"],
                    "train_score": item["train_score"],
                    "eval_score": item["eval_score"],
                    "novelty": item["novelty"],
                    "gap": float(train_score - eval_score),
                }
                for item, train_score, eval_score in zip(buckets, train, eval_scores)
            ],
            "train": train.tolist(),
            "test": eval_scores.tolist(),
            "generalization_gap": float(gaps.mean()),
            "memorization_score": float(train[memorization_mask].mean()),
            "transfer_score": float(eval_scores[generalization_mask].mean()),
            "novelty_correlation": float(np.corrcoef(novelty, eval_scores)[0, 1]),
            "privacy_risk": float((train[memorization_mask] - eval_scores[memorization_mask]).mean()),
            "worst_bucket": worst_bucket["bucket"],
        }
