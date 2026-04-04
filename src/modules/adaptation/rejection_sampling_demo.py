"""Rejection-sampling adaptation demo with candidate pool and threshold logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RejectionSamplingDemo:
    def evaluate(self) -> dict[str, object]:
        threshold = 0.62
        candidates = [
            {"candidate": "safe concise refusal", "reward": 0.74, "accepted": True, "reason": "above_threshold"},
            {"candidate": "overly long refusal", "reward": 0.66, "accepted": True, "reason": "above_threshold"},
            {"candidate": "hedged unsafe answer", "reward": 0.58, "accepted": False, "reason": "below_threshold"},
            {"candidate": "direct unsafe compliance", "reward": 0.41, "accepted": False, "reason": "below_threshold"},
        ]
        candidate_rewards = np.array([item["reward"] for item in candidates], dtype=float)
        accepted_rewards = np.array([item["reward"] for item in candidates if item["accepted"]], dtype=float)
        baseline_loss = 1.35
        adapted_loss = baseline_loss - float(accepted_rewards.mean() * 0.3)
        return {
            "candidate_rewards": candidate_rewards.tolist(),
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "acceptance_rate": float(np.mean([item["accepted"] for item in candidates])),
            "selection_threshold": threshold,
            "candidates": candidates,
        }
