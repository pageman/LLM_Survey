"""Dedicated framework-stack matrix demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameworkStackMatrix:
    def evaluate(self) -> dict[str, object]:
        rows = [
            {"framework": "NumPy-lite", "training": 1, "distributed": 0, "serving": 0, "evaluation": 1},
            {"framework": "JAX", "training": 1, "distributed": 1, "serving": 0, "evaluation": 1},
            {"framework": "PyTorch", "training": 1, "distributed": 1, "serving": 1, "evaluation": 1},
            {"framework": "vLLM", "training": 0, "distributed": 1, "serving": 1, "evaluation": 0},
        ]
        serving_ready = sum(row["serving"] for row in rows)
        distributed_ready = sum(row["distributed"] for row in rows)
        return {
            "num_frameworks": len(rows),
            "serving_ready_fraction": serving_ready / len(rows),
            "distributed_ready_fraction": distributed_ready / len(rows),
            "matrix": rows,
        }
