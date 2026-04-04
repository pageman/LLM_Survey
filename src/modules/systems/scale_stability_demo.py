"""Scale-sensitive stability experiment tied to core normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from src.core import online_softmax, stable_softmax, welford_layer_norm


class ScaleRow(TypedDict):
    scale: float
    naive_finite_fraction: float
    stable_finite_fraction: float
    online_finite_fraction: float
    stable_online_gap: float
    welford_finite_fraction: float


@dataclass
class ScaleStabilityDemo:
    seed: int = 0

    def evaluate(self) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        rows: list[ScaleRow] = []
        for scale in [1.0, 10.0, 100.0, 300.0]:
            logits = rng.standard_normal((4, 8)) * scale
            activations = rng.standard_normal((2, 4, 8)) * scale + (scale * 10.0)

            naive_exp = np.exp(logits)
            naive_denom = np.sum(naive_exp, axis=-1, keepdims=True)
            naive_softmax = np.divide(
                naive_exp,
                naive_denom,
                out=np.full_like(naive_exp, np.nan, dtype=float),
                where=np.isfinite(naive_denom) & (naive_denom != 0.0),
            )
            stable = stable_softmax(logits)
            online = online_softmax(logits)
            welford = welford_layer_norm(activations)

            rows.append(
                {
                    "scale": float(scale),
                    "naive_finite_fraction": float(np.isfinite(naive_softmax).mean()),
                    "stable_finite_fraction": float(np.isfinite(stable).mean()),
                    "online_finite_fraction": float(np.isfinite(online).mean()),
                    "stable_online_gap": float(np.max(np.abs(stable - online))),
                    "welford_finite_fraction": float(np.isfinite(welford).mean()),
                }
            )

        return {
            "rows": rows,
            "worst_naive_finite_fraction": min(row["naive_finite_fraction"] for row in rows),
            "best_stable_finite_fraction": max(row["stable_finite_fraction"] for row in rows),
            "max_stable_online_gap": max(row["stable_online_gap"] for row in rows),
        }
