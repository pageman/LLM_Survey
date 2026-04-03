"""Lite memory-efficient adaptation demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryEfficientAdaptationDemo:
    def evaluate(self) -> dict[str, object]:
        baseline_loss = 1.37
        adapted_loss = 1.09
        trainable_fraction = 0.08
        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "trainable_fraction": trainable_fraction,
            "frozen_fraction": 1.0 - trainable_fraction,
        }
