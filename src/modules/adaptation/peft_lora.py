"""Toy LoRA-style low-rank adaptation on top of a frozen linear map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoRALinearAdapterExperiment:
    input_dim: int
    output_dim: int
    rank: int = 2
    learning_rate: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.W_base = self.rng.standard_normal((self.output_dim, self.input_dim)) * 0.2
        self.A = self.rng.standard_normal((self.rank, self.input_dim)) * 0.01
        self.B = self.rng.standard_normal((self.output_dim, self.rank)) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (self.W_base + self.B @ self.A) @ x

    def mse(self, X: np.ndarray, Y: np.ndarray) -> float:
        preds = np.stack([self.forward(x) for x in X], axis=0)
        return float(np.mean((preds - Y) ** 2))

    def adapt(self, X: np.ndarray, Y: np.ndarray, steps: int = 80) -> dict[str, object]:
        baseline_loss = self.mse(X, Y)
        history = []

        for _ in range(steps):
            grad_A = np.zeros_like(self.A)
            grad_B = np.zeros_like(self.B)
            for x, y in zip(X, Y):
                pred = self.forward(x)
                error = (pred - y)[:, None]
                x_col = x[:, None]
                grad_delta = 2.0 * error @ x_col.T / max(len(X), 1)
                grad_B += grad_delta @ self.A.T
                grad_A += self.B.T @ grad_delta

            self.A -= self.learning_rate * grad_A
            self.B -= self.learning_rate * grad_B
            history.append(self.mse(X, Y))

        adapted_loss = self.mse(X, Y)
        trainable_params = int(self.A.size + self.B.size)
        total_effective_params = int(self.W_base.size + trainable_params)
        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "train_loss_history": history,
            "trainable_fraction": trainable_params / total_effective_params,
        }
