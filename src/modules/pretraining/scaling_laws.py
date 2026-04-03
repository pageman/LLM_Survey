"""Toy scaling law experiments extracted from the donor notebook.

Primary donor notebook:
- 22_scaling_laws.ipynb

This version removes SciPy dependency and fits the power-law exponent with
linear regression in log-log space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def scaling_law_curve(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Simplified power law: y = a * x^(-b)."""
    return a * np.power(x, -b)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = a * x^(-b) via linear regression in log-log space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law fitting requires strictly positive x and y")

    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    a = float(np.exp(intercept))
    b = float(-slope)
    return a, b


@dataclass
class ScalingLawSimulator:
    """Toy language-model simulator for local scaling experiments."""

    vocab_size: int = 100
    noise_scale: float = 0.03
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def train_loss(self, num_params: int, dataset_size: int, num_steps: int) -> float:
        base_loss = math.log(self.vocab_size)
        capacity = math.log(max(num_params, 2)) / 10.0
        param_factor = 1.0 / (1.0 + capacity)
        data_factor = 1.0 / (1.0 + math.log(max(dataset_size, 2)) / 8.0)
        train_factor = math.exp(-num_steps / 1000.0)
        loss = base_loss * param_factor * data_factor * (0.5 + 0.5 * train_factor)
        loss += float(self.rng.normal(0.0, self.noise_scale))
        return max(loss, 0.05)

    def sweep_parameters(
        self,
        param_counts: np.ndarray,
        dataset_size: int,
        num_steps: int,
    ) -> dict[str, object]:
        losses = np.array(
            [self.train_loss(int(n), dataset_size=dataset_size, num_steps=num_steps) for n in param_counts],
            dtype=float,
        )
        a, b = fit_power_law(param_counts, losses)
        return {
            "x": param_counts.tolist(),
            "losses": losses.tolist(),
            "fit": {"a": a, "b": b},
        }

    def sweep_data(
        self,
        dataset_sizes: np.ndarray,
        num_params: int,
        num_steps: int,
    ) -> dict[str, object]:
        losses = np.array(
            [self.train_loss(num_params=num_params, dataset_size=int(d), num_steps=num_steps) for d in dataset_sizes],
            dtype=float,
        )
        a, b = fit_power_law(dataset_sizes, losses)
        return {
            "x": dataset_sizes.tolist(),
            "losses": losses.tolist(),
            "fit": {"a": a, "b": b},
        }

    def compute_optimal_allocation(
        self,
        compute_budgets: np.ndarray,
        num_steps: int,
    ) -> dict[str, object]:
        results = []
        losses = []
        for compute in compute_budgets:
            params = int(max(10, np.sqrt(compute / 6.0)))
            data = int(max(10, compute / (6.0 * params)))
            loss = self.train_loss(num_params=params, dataset_size=data, num_steps=num_steps)
            results.append({"compute": float(compute), "params": params, "data": data, "loss": loss})
            losses.append(loss)

        a, b = fit_power_law(compute_budgets, np.array(losses, dtype=float))
        return {"results": results, "fit": {"a": a, "b": b}}


def run_default_scaling_suite(seed: int = 0) -> dict[str, object]:
    simulator = ScalingLawSimulator(seed=seed)
    param_counts = np.array([1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7], dtype=float)
    dataset_sizes = np.array([1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7], dtype=float)
    compute_budgets = np.array([1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9], dtype=float)

    return {
        "parameter_scaling": simulator.sweep_parameters(param_counts, dataset_size=100_000, num_steps=1000),
        "data_scaling": simulator.sweep_data(dataset_sizes, num_params=1_000_000, num_steps=1000),
        "compute_scaling": simulator.compute_optimal_allocation(compute_budgets, num_steps=1000),
    }
