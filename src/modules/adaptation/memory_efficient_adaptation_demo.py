"""Lite memory-efficient adaptation demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class AdaptationMethodRow(TypedDict):
    method: str
    baseline_loss: float
    adapted_loss: float
    trainable_fraction: float
    memory_multiplier: float


@dataclass
class MemoryEfficientAdaptationDemo:
    def evaluate(self) -> dict[str, object]:
        """Compare full fine-tuning against a memory-efficient adaptation row.

        Returns:
            Dict with the lite memory-efficient row as top-level metrics plus
            `method_rows` for the full comparison table.
        """
        method_rows: list[AdaptationMethodRow] = [
            {
                "method": "full_finetune",
                "baseline_loss": 1.37,
                "adapted_loss": 0.98,
                "trainable_fraction": 1.0,
                "memory_multiplier": 1.0,
            },
            {
                "method": "lora_like",
                "baseline_loss": 1.37,
                "adapted_loss": 1.09,
                "trainable_fraction": 0.08,
                "memory_multiplier": 0.28,
            },
        ]
        baseline_loss = method_rows[1]["baseline_loss"]
        adapted_loss = method_rows[1]["adapted_loss"]
        trainable_fraction = method_rows[1]["trainable_fraction"]
        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "trainable_fraction": trainable_fraction,
            "frozen_fraction": 1.0 - trainable_fraction,
            "memory_saving": 1.0 - method_rows[1]["memory_multiplier"],
            "method_rows": method_rows,
        }
