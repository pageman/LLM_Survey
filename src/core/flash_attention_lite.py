"""Educational blockwise attention to approximate FlashAttention-style tiling."""

from __future__ import annotations

import math

import numpy as np

from .types import Array2D, FloatArray, MaskArray


def flash_attention_lite(
    query: Array2D,
    key: Array2D,
    value: Array2D,
    mask: MaskArray | None = None,
    block_size: int = 4,
) -> tuple[Array2D, dict[str, float | int]]:
    """Compute blockwise attention without materializing one full score matrix.

    Args:
        query: Query matrix shaped ``(seq_q, dim)``.
        key: Key matrix shaped ``(seq_k, dim)``.
        value: Value matrix shaped ``(seq_k, dim_v)``.
        mask: Optional additive mask shaped ``(seq_q, seq_k)``.
        block_size: Key/value tile size.

    Returns:
        A tuple of ``(output, diagnostics)`` where ``output`` has shape
        ``(seq_q, dim_v)`` and diagnostics summarize the tiling behavior.
    """
    query = np.asarray(query, dtype=float)
    key = np.asarray(key, dtype=float)
    value = np.asarray(value, dtype=float)
    if query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("flash_attention_lite expects rank-2 query/key/value arrays")
    if query.shape[1] != key.shape[1]:
        raise ValueError("query and key width must match")
    if key.shape[0] != value.shape[0]:
        raise ValueError("key and value sequence length must match")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if mask is not None and mask.shape != (query.shape[0], key.shape[0]):
        raise ValueError("mask must have shape (query_len, key_len)")

    scale = 1.0 / math.sqrt(query.shape[-1])
    out = np.zeros((query.shape[0], value.shape[1]), dtype=float)
    row_max = np.full((query.shape[0],), -np.inf, dtype=float)
    row_sum = np.zeros((query.shape[0],), dtype=float)

    for start in range(0, key.shape[0], block_size):
        stop = min(start + block_size, key.shape[0])
        key_block = key[start:stop]
        value_block = value[start:stop]
        scores = np.einsum("qd,kd->qk", query, key_block) * scale
        if mask is not None:
            scores = scores + (mask[:, start:stop] * -1e9)

        block_max = np.max(scores, axis=-1)
        next_row_max = np.maximum(row_max, block_max)
        exp_scores = np.exp(scores - np.expand_dims(next_row_max, axis=-1))
        row_rescale = np.exp(row_max - next_row_max)

        out = out * np.expand_dims(row_rescale, axis=-1)
        out = out + np.einsum("qk,kd->qd", exp_scores, value_block)
        row_sum = row_sum * row_rescale + np.sum(exp_scores, axis=-1)
        row_max = next_row_max

    output = out / np.expand_dims(row_sum, axis=-1)
    diagnostics: dict[str, float | int] = {
        "query_len": int(query.shape[0]),
        "key_len": int(key.shape[0]),
        "block_size": int(block_size),
        "num_blocks": int(math.ceil(key.shape[0] / block_size)),
        "materialized_score_elements": int(query.shape[0] * min(block_size, key.shape[0])),
    }
    return output, diagnostics
