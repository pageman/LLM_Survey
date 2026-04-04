"""Compare numerically stable helpers against naive implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import layer_norm, online_softmax, stable_softmax, welford_layer_norm


@dataclass
class NumericStabilityDemo:
    seed: int = 0

    def evaluate(self) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        logits = rng.standard_normal((4, 8)) * 200.0
        activations = rng.standard_normal((3, 6, 8)) * 1e4 + 1e6

        naive_exp = np.exp(logits)
        naive_denominator = np.sum(naive_exp, axis=-1, keepdims=True)
        naive_softmax = np.divide(
            naive_exp,
            naive_denominator,
            out=np.full_like(naive_exp, np.nan, dtype=float),
            where=np.isfinite(naive_denominator) & (naive_denominator != 0.0),
        )
        stable = stable_softmax(logits)
        online = online_softmax(logits)

        naive_mean = np.mean(activations, axis=-1, keepdims=True)
        naive_var = np.var(activations, axis=-1, keepdims=True)
        naive_layer = (activations - naive_mean) / np.sqrt(naive_var + 1e-6)
        welford = welford_layer_norm(activations)
        reference = layer_norm(activations)

        return {
            "naive_softmax_finite_fraction": float(np.isfinite(naive_softmax).mean()),
            "stable_softmax_finite_fraction": float(np.isfinite(stable).mean()),
            "online_softmax_finite_fraction": float(np.isfinite(online).mean()),
            "stable_online_gap": float(np.max(np.abs(stable - online))),
            "naive_layer_norm_finite_fraction": float(np.isfinite(naive_layer).mean()),
            "welford_layer_norm_finite_fraction": float(np.isfinite(welford).mean()),
            "welford_reference_gap": float(np.max(np.abs(welford - reference))),
        }
