"""Minimal transformer utilities built on top of extracted attention primitives.

Primary donor notebook:
- 13_attention_is_all_you_need.ipynb
"""

from __future__ import annotations

import numpy as np

from .attention import MultiHeadAttention


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding."""
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (dims // 2)) / np.maximum(d_model, 1))
    angles = positions * angle_rates

    encoding = np.zeros((seq_len, d_model), dtype=float)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def feed_forward(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    hidden = np.maximum(0.0, (x @ W1.T) + b1)
    return (hidden @ W2.T) + b2


class TransformerBlock:
    """Pre-norm-style transformer block for educational experiments."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rng: np.random.Generator | None = None,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rng = rng or np.random.default_rng()

        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, rng=self.rng)
        scale = 0.1
        self.W1 = self.rng.standard_normal((d_ff, d_model)) * scale
        self.b1 = np.zeros((d_ff,), dtype=float)
        self.W2 = self.rng.standard_normal((d_model, d_ff)) * scale
        self.b2 = np.zeros((d_model,), dtype=float)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        normed = layer_norm(x)
        attn_out = self.attention.forward(normed, normed, normed, mask=mask)
        x = x + attn_out

        ff_in = layer_norm(x)
        ff_out = feed_forward(ff_in, self.W1, self.b1, self.W2, self.b2)
        x = x + ff_out

        if self.attention.attention_weights is None:
            raise RuntimeError("attention weights were not recorded")
        return x, self.attention.attention_weights
