"""Lite mixture-of-experts routing demo with explicit expert specialization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MoEDemo:
    num_experts: int = 3

    def __post_init__(self) -> None:
        self.router = np.array(
            [
                [1.2, -0.4, -0.6],
                [-0.5, 1.1, -0.3],
                [-0.4, -0.2, 1.3],
            ],
            dtype=float,
        )
        self.experts = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.7, 0.7],
            ],
            dtype=float,
        )

    def evaluate(self, inputs: np.ndarray | None = None) -> dict[str, object]:
        inputs = inputs if inputs is not None else np.array(
            [
                [1.0, 0.1, 0.0],
                [0.1, 1.0, 0.0],
                [0.2, 0.1, 1.0],
                [0.8, 0.2, 0.1],
            ],
            dtype=float,
        )
        logits = np.einsum("sd,de->se", inputs, self.router, optimize=True)
        logits = logits - logits.max(axis=1, keepdims=True)
        routing = np.exp(logits)
        routing /= routing.sum(axis=1, keepdims=True)
        top_expert = np.argmax(routing, axis=1)
        outputs = np.einsum("se,ed->sd", routing, self.experts, optimize=True)
        load = routing.mean(axis=0)
        normalized_entropy = float(-(routing * np.log(routing + 1e-9)).sum(axis=1).mean() / np.log(self.num_experts))

        return {
            "routing": routing.round(4).tolist(),
            "top_expert": top_expert.astype(int).tolist(),
            "expert_outputs": outputs.round(4).tolist(),
            "load_balance": float(load.min() / load.max()),
            "expert_specialization": float(1.0 - normalized_entropy),
        }
