"""PPO-style RLHF toy demo with rollout, clipping, and acceptance traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PPORLFHToy:
    def evaluate(self) -> dict[str, object]:
        rollouts = [
            {"step": 0, "reward": 0.34, "kl": 0.02, "advantage": 0.11, "ratio": 1.04, "accepted": True},
            {"step": 1, "reward": 0.49, "kl": 0.05, "advantage": 0.17, "ratio": 1.12, "accepted": True},
            {"step": 2, "reward": 0.58, "kl": 0.08, "advantage": 0.21, "ratio": 1.27, "accepted": False},
            {"step": 3, "reward": 0.62, "kl": 0.11, "advantage": 0.24, "ratio": 1.18, "accepted": True},
        ]
        reward = np.array([item["reward"] for item in rollouts], dtype=float)
        kl = np.array([item["kl"] for item in rollouts], dtype=float)
        advantage = np.array([item["advantage"] for item in rollouts], dtype=float)
        ratio = np.array([item["ratio"] for item in rollouts], dtype=float)
        clipped_ratio = np.clip(ratio, 0.8, 1.2)
        surrogate = np.minimum(ratio * advantage, clipped_ratio * advantage)
        accepted = np.array([float(item["accepted"]) for item in rollouts], dtype=float)
        baseline_loss = 1.41
        policy_gain = float((surrogate - 0.4 * kl).mean())
        adapted_loss = baseline_loss - policy_gain
        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "reward_curve": reward.tolist(),
            "kl_curve": kl.tolist(),
            "surrogate_objective": surrogate.tolist(),
            "clipped_ratio": clipped_ratio.tolist(),
            "acceptance_rate": float(accepted.mean()),
            "mean_policy_delta": float(np.abs(clipped_ratio - 1.0).mean()),
            "rollouts": [
                {
                    **item,
                    "clipped_ratio": float(clipped),
                    "surrogate": float(obj),
                }
                for item, clipped, obj in zip(rollouts, clipped_ratio, surrogate)
            ],
        }
