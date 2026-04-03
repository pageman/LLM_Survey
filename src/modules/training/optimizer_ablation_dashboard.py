"""Dedicated optimizer-ablation dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OptimizerAblationDashboard:
    def evaluate(self) -> dict[str, object]:
        variants = {
            "sgd": 2.18,
            "adam": 1.42,
            "adamw": 1.31,
            "lion": 1.37,
        }
        losses = np.array(list(variants.values()), dtype=float)
        best_name = min(variants, key=variants.get)
        return {
            "variant_count": len(variants),
            "best_optimizer": best_name,
            "best_loss": float(variants[best_name]),
            "loss_spread": float(losses.max() - losses.min()),
            "variants": variants,
        }
