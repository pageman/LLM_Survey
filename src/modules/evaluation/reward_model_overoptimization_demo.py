"""Reward-model overoptimization demo with proxy-target divergence traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardModelOveroptimizationDemo:
    def evaluate(self) -> dict[str, object]:
        steps = [
            {"step": 0, "reward": 0.52, "factuality": 0.70, "policy_confidence": 0.56},
            {"step": 1, "reward": 0.64, "factuality": 0.63, "policy_confidence": 0.67},
            {"step": 2, "reward": 0.76, "factuality": 0.56, "policy_confidence": 0.81},
            {"step": 3, "reward": 0.83, "factuality": 0.49, "policy_confidence": 0.88},
        ]
        reward = np.array([item["reward"] for item in steps], dtype=float)
        factuality = np.array([item["factuality"] for item in steps], dtype=float)
        proxy_gap = reward - factuality
        return {
            "reward": reward.tolist(),
            "factuality": factuality.tolist(),
            "reward_factuality_correlation": float(np.corrcoef(reward, factuality)[0, 1]),
            "overoptimization_gap": float(proxy_gap.mean()),
            "max_proxy_gap": float(proxy_gap.max()),
            "steps": [
                {
                    **item,
                    "proxy_gap": float(gap),
                }
                for item, gap in zip(steps, proxy_gap)
            ],
        }
