"""Attention primitives extracted and normalized from donor notebooks.

Primary donor notebooks:
- 13_attention_is_all_you_need.ipynb
- 14_bahdanau_attention.ipynb
"""

from __future__ import annotations

import numpy as np

from .protocols import AttentionModule
from .ops import glorot_scale, stable_softmax
from .types import Array1D, Array2D, Array3D, FloatArray, MaskArray


def softmax(x: FloatArray, axis: int = -1) -> FloatArray:
    """Numerically stable softmax."""
    return stable_softmax(x, axis=axis)


def create_causal_mask(seq_len: int) -> MaskArray:
    """Return an additive mask where 1 blocks attention to future positions."""
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    return np.triu(np.ones((seq_len, seq_len), dtype=float), k=1)


def scaled_dot_product_attention(
    query: Array2D | Array3D,
    key: Array2D | Array3D,
    value: Array2D | Array3D,
    mask: MaskArray | None = None,
) -> tuple[Array2D | Array3D, FloatArray]:
    """Compute scaled dot-product attention for 2D or batched 3D arrays.

    Args:
        query: ``(q_len, d_k)`` or ``(batch, q_len, d_k)``
        key: ``(k_len, d_k)`` or ``(batch, k_len, d_k)``
        value: ``(k_len, d_v)`` or ``(batch, k_len, d_v)``
        mask: Optional additive mask shaped ``(q_len, k_len)`` or ``(batch, q_len, k_len)``
            where non-zero entries are blocked.

    Returns:
        Tuple of ``(output, attention_weights)`` with the same leading dimensions as ``query``.

    Raises:
        ValueError: If dimensions do not line up or if an empty key sequence is provided.
    """
    query = np.asarray(query, dtype=float)
    key = np.asarray(key, dtype=float)
    value = np.asarray(value, dtype=float)
    if query.ndim not in (2, 3) or key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("query, key, and value must all be 2D or all be 3D arrays")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must share the same feature dimension")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value must share the same sequence length")
    if key.shape[-2] == 0:
        raise ValueError("attention requires a non-empty key/value sequence")

    d_k = query.shape[-1]
    if query.ndim == 2:
        scores = np.einsum("qd,kd->qk", query, key, optimize=True) / np.sqrt(d_k)
    else:
        scores = np.einsum("bqd,bkd->bqk", query, key, optimize=True) / np.sqrt(d_k)

    if mask is not None:
        mask = np.asarray(mask, dtype=float)
        scores = scores + (mask * -1e9)

    attention_weights = softmax(scores, axis=-1)
    if query.ndim == 2:
        output = np.einsum("qk,kd->qd", attention_weights, value, optimize=True)
    else:
        output = np.einsum("bqk,bkd->bqd", attention_weights, value, optimize=True)
    return output, attention_weights


class MultiHeadAttention(AttentionModule):
    """Minimal NumPy multi-head attention.

    The API mirrors the donor notebook but standardizes storage and shapes.
    Inputs are expected to be shaped ``(seq_len, d_model)``.
    """

    def __init__(self, d_model: int, num_heads: int, rng: np.random.Generator | None = None):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rng = rng or np.random.default_rng()

        scale = glorot_scale(d_model, d_model)
        self.W_q = self.rng.standard_normal((d_model, d_model)) * scale
        self.W_k = self.rng.standard_normal((d_model, d_model)) * scale
        self.W_v = self.rng.standard_normal((d_model, d_model)) * scale
        self.W_o = self.rng.standard_normal((d_model, d_model)) * scale
        self.attention_weights: FloatArray | None = None

    def split_heads(self, x: Array2D) -> Array3D:
        if x.ndim != 2 or x.shape[-1] != self.d_model:
            raise ValueError("expected x with shape (seq_len, d_model)")
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

    def combine_heads(self, x: Array3D) -> Array2D:
        seq_len = x.shape[1]
        return x.transpose(1, 0, 2).reshape(seq_len, self.d_model)

    def forward(
        self,
        query: Array2D,
        key: Array2D,
        value: Array2D,
        mask: MaskArray | None = None,
    ) -> Array2D:
        """Apply sequence-major multi-head attention over ``(Seq, Dim)`` arrays."""
        if query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
            raise ValueError("multi-head attention expects 2D arrays")
        if query.shape[-1] != self.d_model or key.shape[-1] != self.d_model or value.shape[-1] != self.d_model:
            raise ValueError("query, key, and value must all have trailing dimension d_model")
        query_proj = query @ self.W_q.T
        key_proj = key @ self.W_k.T
        value_proj = value @ self.W_v.T

        query_heads = self.split_heads(query_proj)
        key_heads = self.split_heads(key_proj)
        value_heads = self.split_heads(value_proj)

        heads, self.attention_weights = scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            mask=mask,
        )
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
        if decoder_hidden_size <= 0 or annotation_size <= 0:
            raise ValueError("decoder_hidden_size and annotation_size must be positive")
        self.decoder_hidden_size = decoder_hidden_size
        self.annotation_size = annotation_size
        self.rng = rng or np.random.default_rng()

        scale = glorot_scale(decoder_hidden_size, annotation_size)
        self.W_a = self.rng.standard_normal((decoder_hidden_size, decoder_hidden_size)) * scale
        self.U_a = self.rng.standard_normal((decoder_hidden_size, annotation_size)) * scale
        self.v_a = self.rng.standard_normal((1, decoder_hidden_size)) * scale

    def forward(
        self,
        decoder_hidden: Array2D,
        encoder_annotations: list[Array2D] | Array2D,
    ) -> tuple[Array2D, Array1D]:
        decoder_hidden = np.asarray(decoder_hidden, dtype=float)
        if decoder_hidden.shape != (self.decoder_hidden_size, 1):
            raise ValueError("decoder_hidden must have shape (decoder_hidden_size, 1)")
        if isinstance(encoder_annotations, np.ndarray):
            if encoder_annotations.ndim != 2 or encoder_annotations.shape[1] != self.annotation_size:
                raise ValueError("encoder_annotations must have shape (seq_len, annotation_size)")
            annotations_matrix = np.asarray(encoder_annotations, dtype=float)
        else:
            if not encoder_annotations:
                raise ValueError("encoder_annotations must be non-empty")
            annotations_matrix = np.concatenate(
                [np.asarray(annotation, dtype=float).reshape(1, self.annotation_size) for annotation in encoder_annotations],
                axis=0,
            )
        projected_decoder = np.einsum("hd,dn->nh", self.W_a, decoder_hidden, optimize=True)
        projected_annotations = np.einsum("sa,ha->sh", annotations_matrix, self.U_a, optimize=True)
        energies = np.tanh(projected_annotations + projected_decoder)
        scores = np.einsum("sh,oh->s", energies, self.v_a, optimize=True)
        attention_weights = softmax(scores, axis=-1)
        context = np.einsum("s,sa->a", attention_weights, annotations_matrix, optimize=True).reshape(
            self.annotation_size,
            1,
        )
        return context, attention_weights
