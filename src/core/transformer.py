"""Minimal transformer utilities built on top of extracted attention primitives.

Primary donor notebook:
- 13_attention_is_all_you_need.ipynb
"""

from __future__ import annotations

import numpy as np

from .attention import MultiHeadAttention
from .ops import feed_forward, glorot_scale, layer_norm
from .types import Array2D, FloatArray, MaskArray


def positional_encoding(seq_len: int, d_model: int) -> Array2D:
    """Return sinusoidal positional encoding with shape ``(seq_len, d_model)``."""
    if seq_len < 0 or d_model <= 0:
        raise ValueError("seq_len must be non-negative and d_model must be positive")
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (dims // 2)) / np.maximum(d_model, 1))
    angles = positions * angle_rates

    encoding = np.zeros((seq_len, d_model), dtype=float)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding


class TransformerBlock:
    """Pre-norm-style transformer block for educational experiments.

    Args:
        d_model: Model width.
        num_heads: Number of attention heads.
        d_ff: Hidden size of the feed-forward block.
        rng: Optional NumPy random generator.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rng: np.random.Generator | None = None,
    ):
        if d_model <= 0 or num_heads <= 0 or d_ff <= 0:
            raise ValueError("d_model, num_heads, and d_ff must be positive")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rng = rng or np.random.default_rng()

        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, rng=self.rng)
        scale = glorot_scale(d_model, d_ff)
        self.W1 = self.rng.standard_normal((d_ff, d_model)) * scale
        self.b1 = np.zeros((d_ff,), dtype=float)
        self.W2 = self.rng.standard_normal((d_model, d_ff)) * scale
        self.b2 = np.zeros((d_model,), dtype=float)

    def forward(self, x: Array2D, mask: MaskArray | None = None) -> tuple[Array2D, FloatArray]:
        """Run one transformer block over ``x`` shaped ``(seq_len, d_model)``."""
        if x.ndim != 2 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape (seq_len, d_model)")
        normed = layer_norm(x)
        attn_out = self.attention.forward(normed, normed, normed, mask=mask)
        x = x + attn_out

        ff_in = layer_norm(x)
        ff_out = feed_forward(ff_in, self.W1, self.b1, self.W2, self.b2)
        x = x + ff_out

        if self.attention.attention_weights is None:
            raise RuntimeError("attention weights were not recorded")
        return x, self.attention.attention_weights
