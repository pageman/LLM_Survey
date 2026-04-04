"""Lite encoder-decoder demo with explicit cross-attention behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EncoderDecoderDemo:
    d_model: int = 4

    def __post_init__(self) -> None:
        self.token_embeddings = {
            "translate": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            "hello": np.array([0.0, 1.0, 0.0, 0.0], dtype=float),
            "world": np.array([0.0, 0.0, 1.0, 0.0], dtype=float),
            "hola": np.array([0.0, 0.9, 0.1, 0.0], dtype=float),
            "mundo": np.array([0.0, 0.1, 0.9, 0.0], dtype=float),
        }

    def _embed(self, tokens: list[str]) -> np.ndarray:
        return np.vstack([self.token_embeddings[token] for token in tokens])

    def evaluate(self, source_tokens: list[str] | None = None, target_queries: list[str] | None = None) -> dict[str, object]:
        source_tokens = source_tokens or ["hello", "world"]
        target_queries = target_queries or ["hola", "mundo"]

        encoder_states = self._embed(source_tokens)
        decoder_queries = self._embed(target_queries)
        scores = np.einsum("qd,kd->qk", decoder_queries, encoder_states, optimize=True)
        scores = scores - scores.max(axis=1, keepdims=True)
        cross_attention = np.exp(scores)
        cross_attention /= cross_attention.sum(axis=1, keepdims=True)
        context = np.einsum("qk,kd->qd", cross_attention, encoder_states, optimize=True)

        source_positions = np.argmax(encoder_states[:, 1:3], axis=1)
        target_positions = np.argmax(context[:, 1:3], axis=1)
        copy_accuracy = float(np.mean(source_positions == target_positions))

        return {
            "source_tokens": source_tokens,
            "target_queries": target_queries,
            "cross_attention": cross_attention.round(4).tolist(),
            "context_vectors": context.round(4).tolist(),
            "cross_attention_focus": float(cross_attention.max(axis=1).mean()),
            "copy_accuracy": copy_accuracy,
        }
