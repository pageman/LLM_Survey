"""Toy LoRA-style low-rank adaptation on top of a frozen linear map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import Array1D, Array2D

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

    def forward(self, x: Array1D) -> Array1D:
        """Project one input vector through the frozen-plus-LoRA linear map."""
        if x.ndim != 1 or x.shape[0] != self.input_dim:
            raise ValueError("x must have shape (input_dim,)")
        delta = self.B @ self.A
        return (self.W_base + delta) @ x

    def mse(self, X: Array2D, Y: Array2D) -> float:
        """Return the mean squared error over a matrix of row-wise examples."""
        preds = np.stack([self.forward(x) for x in X], axis=0)
        return float(np.mean((preds - Y) ** 2))

    def adapt(self, X: Array2D, Y: Array2D, steps: int = 80) -> dict[str, object]:
        """Fit the low-rank factors while keeping the base weight frozen."""
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be rank-2 arrays")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")
        if X.shape[1] != self.input_dim or Y.shape[1] != self.output_dim:
            raise ValueError("X and Y must match the adapter input/output dimensions")
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
