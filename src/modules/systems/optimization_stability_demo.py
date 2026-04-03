"""Toy optimization stability demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizationStabilityDemo:
    def evaluate(self) -> dict[str, object]:
        unclipped_grad_norm = 8.5
        clipped_grad_norm = 1.0
        unstable_loss = 2.4
        stable_loss = 1.6
        return {
            "unclipped_grad_norm": unclipped_grad_norm,
            "clipped_grad_norm": clipped_grad_norm,
            "unstable_loss": unstable_loss,
            "stable_loss": stable_loss,
            "stability_gain": unstable_loss - stable_loss,
        }
