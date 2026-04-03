"""Attention primitives extracted and normalized from donor notebooks.

Primary donor notebooks:
- 13_attention_is_all_you_need.ipynb
- 14_bahdanau_attention.ipynb
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """Return an additive mask where 1 blocks attention to future positions."""
    return np.triu(np.ones((seq_len, seq_len), dtype=float), k=1)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention for 2D arrays.

    Args:
        query: (q_len, d_k)
        key: (k_len, d_k)
        value: (k_len, d_v)
        mask: optional (q_len, k_len) additive mask where non-zero entries are blocked
    """
    d_k = query.shape[-1]
    scores = query @ key.T / np.sqrt(d_k)

    if mask is not None:
        scores = scores + (mask * -1e9)

    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ value
    return output, attention_weights


class MultiHeadAttention:
    """Minimal NumPy multi-head attention.

    The API mirrors the donor notebook but standardizes storage and shapes.
    """

    def __init__(self, d_model: int, num_heads: int, rng: np.random.Generator | None = None):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rng = rng or np.random.default_rng()

        scale = 0.1
        self.W_q = self.rng.standard_normal((d_model, d_model)) * scale
        self.W_k = self.rng.standard_normal((d_model, d_model)) * scale
        self.W_v = self.rng.standard_normal((d_model, d_model)) * scale
        self.W_o = self.rng.standard_normal((d_model, d_model)) * scale
        self.attention_weights: np.ndarray | None = None

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[1]
        return x.transpose(1, 0, 2).reshape(seq_len, self.d_model)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        query_proj = query @ self.W_q.T
        key_proj = key @ self.W_k.T
        value_proj = value @ self.W_v.T

        query_heads = self.split_heads(query_proj)
        key_heads = self.split_heads(key_proj)
        value_heads = self.split_heads(value_proj)

        head_outputs = []
        head_attentions = []
        for head_idx in range(self.num_heads):
            out, weights = scaled_dot_product_attention(
                query_heads[head_idx],
                key_heads[head_idx],
                value_heads[head_idx],
                mask=mask,
            )
            head_outputs.append(out)
            head_attentions.append(weights)

        heads = np.stack(head_outputs, axis=0)
        self.attention_weights = np.stack(head_attentions, axis=0)
        return self.combine_heads(heads) @ self.W_o.T


class BahdanauAttention:
    """Additive attention over encoder annotations.

    Args:
        decoder_hidden_size: size of decoder hidden state
        annotation_size: size of encoder annotations
    """

    def __init__(
        self,
        decoder_hidden_size: int,
        annotation_size: int,
        rng: np.random.Generator | None = None,
    ):
        self.decoder_hidden_size = decoder_hidden_size
        self.annotation_size = annotation_size
        self.rng = rng or np.random.default_rng()

        scale = 0.01
        self.W_a = self.rng.standard_normal((decoder_hidden_size, decoder_hidden_size)) * scale
        self.U_a = self.rng.standard_normal((decoder_hidden_size, annotation_size)) * scale
        self.v_a = self.rng.standard_normal((1, decoder_hidden_size)) * scale

    def forward(
        self,
        decoder_hidden: np.ndarray,
        encoder_annotations: list[np.ndarray] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(encoder_annotations, np.ndarray):
            annotations = [encoder_annotations[t : t + 1].T for t in range(encoder_annotations.shape[0])]
        else:
            annotations = encoder_annotations

        scores = []
        for annotation in annotations:
            score = self.v_a @ np.tanh((self.W_a @ decoder_hidden) + (self.U_a @ annotation))
            scores.append(score[0, 0])

        attention_weights = softmax(np.array(scores), axis=-1)
        context = sum(alpha * annotation for alpha, annotation in zip(attention_weights, annotations))
        return context, attention_weights
