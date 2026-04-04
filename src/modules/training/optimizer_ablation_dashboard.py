"""Optimizer-ablation demo with loss trajectories and stability signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OptimizerAblationDashboard:
    def evaluate(self) -> dict[str, object]:
        trajectories = {
            "sgd": [2.48, 2.33, 2.17, 2.05, 1.97],
            "adam": [2.31, 1.96, 1.69, 1.53, 1.42],
            "adamw": [2.29, 1.89, 1.58, 1.41, 1.31],
            "lion": [2.34, 1.94, 1.63, 1.48, 1.37],
        }
        final_losses = {name: values[-1] for name, values in trajectories.items()}
        losses = np.array(list(final_losses.values()), dtype=float)
        best_name = min(final_losses, key=final_losses.get)
        variants = []
        for name, values in trajectories.items():
            curve = np.array(values, dtype=float)
            improvements = curve[:-1] - curve[1:]
            variants.append(
                {
                    "optimizer": name,
                    "loss_curve": values,
                    "final_loss": float(curve[-1]),
                    "best_step": int(np.argmin(curve)),
                    "mean_step_improvement": float(improvements.mean()),
                    "stability_score": float(1.0 / (1.0 + np.std(improvements))),
                }
            )
        return {
            "variant_count": len(variants),
            "best_optimizer": best_name,
            "best_loss": float(final_losses[best_name]),
            "loss_spread": float(losses.max() - losses.min()),
            "best_stability": max(variants, key=lambda item: item["stability_score"])["optimizer"],
            "variants": variants,
        }
