"""Lite objective-mixture demo with explicit grid search over objective weights."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class ObjectiveMixtureDemo:
    def evaluate(self) -> dict[str, object]:
        # Columns: next-token, denoising, contrastive retrieval.
        candidate_losses = np.array(
            [
                [1.26, 1.12, 1.08],
                [1.11, 0.99, 0.95],
                [1.03, 0.92, 0.88],
                [1.08, 0.9, 0.84],
            ],
            dtype=float,
        )

        weight_grid = []
        for a, b in product([0.2, 0.4, 0.6], repeat=2):
            c = 1.0 - a - b
            if c < 0.0:
                continue
            weight_grid.append([a, b, c])
        weights = np.array(weight_grid, dtype=float)

        mixed_loss = candidate_losses.mean(axis=0)[None, :] @ weights.T
        mixed_loss = mixed_loss.reshape(-1)
        best_idx = int(np.argmin(mixed_loss))
        best_weights = weights[best_idx]
        baseline_loss = float(candidate_losses.mean(axis=0)[0])
        best_loss = float(mixed_loss[best_idx])

        return {
            "candidate_losses": candidate_losses.round(4).tolist(),
            "weight_grid": weights.round(4).tolist(),
            "best_weights": best_weights.round(4).tolist(),
            "best_mixture_loss": best_loss,
            "mixture_gain": baseline_loss - best_loss,
        }
