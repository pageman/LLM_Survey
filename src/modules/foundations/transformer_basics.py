"""Decoder-only transformer demo built on the extracted transformer block."""

from __future__ import annotations

import numpy as np

from src.core import TransformerBlock, create_causal_mask, positional_encoding, softmax


class DecoderOnlyTransformerDemo:
    """A tiny decoder-only transformer for forward-pass demonstrations."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int = 1,
        max_seq_len: int = 64,
        seed: int = 0,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.rng = np.random.default_rng(seed)

        scale = 0.1
        self.token_embedding = self.rng.standard_normal((vocab_size, d_model)) * scale
        self.output_projection = self.rng.standard_normal((vocab_size, d_model)) * scale
        self.blocks = [
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rng=self.rng)
            for _ in range(num_layers)
        ]

    def embed(self, tokens: list[int]) -> np.ndarray:
        seq_len = len(tokens)
        if seq_len > self.max_seq_len:
            raise ValueError("sequence exceeds configured max_seq_len")
        x = self.token_embedding[np.array(tokens)]
        return x + positional_encoding(seq_len, self.d_model)

    def forward(self, tokens: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
        x = self.embed(tokens)
        mask = create_causal_mask(len(tokens))
        attention_maps = []
        for block in self.blocks:
            x, weights = block.forward(x, mask=mask)
            attention_maps.append(weights)
        logits = x @ self.output_projection.T
        return logits, attention_maps

    def predict_next_distribution(self, tokens: list[int]) -> np.ndarray:
        logits, _ = self.forward(tokens)
        return softmax(logits[-1], axis=-1)
