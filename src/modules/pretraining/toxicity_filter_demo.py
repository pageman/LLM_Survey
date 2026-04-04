"""Toxicity-filter demo with threshold sweep and retention tradeoffs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToxicityFilterDemo:
    def evaluate(self) -> dict[str, object]:
        toxicity = np.array([0.92, 0.74, 0.31, 0.16, 0.07, 0.03], dtype=float)
        thresholds = [0.2, 0.35, 0.5]
        sweep = []
        for threshold in thresholds:
            kept = toxicity < threshold
            sweep.append(
                {
                    "threshold": threshold,
                    "retention_rate": float(kept.mean()),
                    "toxicity_reduction": float(toxicity.mean() - toxicity[kept].mean()),
                    "proxy_precision": float(0.82 - threshold * 0.35),
                    "proxy_recall": float(0.58 + threshold * 0.55),
                }
            )
        kept = toxicity < 0.35
        return {
            "raw_toxicity": toxicity.tolist(),
            "retained_mask": kept.astype(int).tolist(),
            "retention_rate": float(kept.mean()),
            "toxicity_reduction": float(toxicity.mean() - toxicity[kept].mean()),
            "threshold_sweep": sweep,
        }
