"""Minimal RNN LM wrapper around the reusable vanilla RNN core."""

from __future__ import annotations

import numpy as np

from src.core import VanillaRNNLanguageModel


class RNNLanguageModel:
    """Token-level RNN LM with a small local training step API."""

    def __init__(self, vocab_size: int, hidden_size: int, learning_rate: float = 0.1, seed: int = 0):
        self.model = VanillaRNNLanguageModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            rng=np.random.default_rng(seed),
        )
        self.learning_rate = learning_rate

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    def train_step(self, inputs: list[int], targets: list[int]) -> float:
        xs, hs, _, ps = self.model.forward(inputs)
        loss = self.model.loss(ps, targets)
        grads = self.model.backward(xs, hs, ps, targets)

        self.model.Wxh -= self.learning_rate * grads["Wxh"]
        self.model.Whh -= self.learning_rate * grads["Whh"]
        self.model.Why -= self.learning_rate * grads["Why"]
        self.model.bh -= self.learning_rate * grads["bh"]
        self.model.by -= self.learning_rate * grads["by"]
        return loss

    def evaluate_loss(self, inputs: list[int], targets: list[int]) -> float:
        _, _, _, ps = self.model.forward(inputs)
        return self.model.loss(ps, targets)

    def sample(self, seed_token: int, length: int) -> list[int]:
        return self.model.sample(seed_ix=seed_token, n=length)

    def get_params(self) -> dict[str, np.ndarray]:
        return {
            "Wxh": self.model.Wxh.copy(),
            "Whh": self.model.Whh.copy(),
            "Why": self.model.Why.copy(),
            "bh": self.model.bh.copy(),
            "by": self.model.by.copy(),
        }

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        self.model.Wxh = params["Wxh"].copy()
        self.model.Whh = params["Whh"].copy()
        self.model.Why = params["Why"].copy()
        self.model.bh = params["bh"].copy()
        self.model.by = params["by"].copy()
