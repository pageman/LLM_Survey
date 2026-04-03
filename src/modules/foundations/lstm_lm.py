"""Minimal LSTM LM wrapper for sequence demonstrations."""

from __future__ import annotations

import numpy as np

from src.core import LSTMSequenceModel


class LSTMLanguageModel:
    """Small LSTM LM facade using one-hot token vectors as inputs."""

    def __init__(self, vocab_size: int, hidden_size: int, seed: int = 0):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rng = np.random.default_rng(seed)
        self.model = LSTMSequenceModel(
            input_size=vocab_size,
            hidden_size=hidden_size,
            output_size=vocab_size,
            rng=self.rng,
        )

    def encode_tokens(self, tokens: list[int]) -> list[np.ndarray]:
        encoded = []
        for token in tokens:
            vector = np.zeros((self.vocab_size, 1), dtype=float)
            vector[token, 0] = 1.0
            encoded.append(vector)
        return encoded

    def forward(self, tokens: list[int]) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], dict[str, list[np.ndarray]]]:
        return self.model.forward(self.encode_tokens(tokens))

    def predict_next_distribution(self, tokens: list[int]) -> np.ndarray:
        logits, _, _, _ = self.forward(tokens)
        logits = logits[:, 0]
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    def sample_next(self, tokens: list[int]) -> int:
        probs = self.predict_next_distribution(tokens)
        return int(self.rng.choice(self.vocab_size, p=probs))
