"""Shared NumPy ops used across attention and transformer primitives."""

from __future__ import annotations

import math

import numpy as np

from .types import Array1D, Array2D, Array3D, FloatArray


def stable_softmax(x: FloatArray, axis: int = -1) -> FloatArray:
    """Return a numerically stable softmax over the given axis.

    Args:
        x: Input array with arbitrary leading dimensions.
        axis: Axis over which to normalize.

    Returns:
        Array with the same shape as ``x`` where values sum to one along ``axis``.
    """
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def online_softmax(x: FloatArray, axis: int = -1) -> FloatArray:
    """Return softmax via an online max/sum-exp pass.

    This mirrors the numerically stable streaming logic used in more advanced
    attention implementations while staying fully NumPy-only.

    Args:
        x: Input array with arbitrary leading dimensions.
        axis: Axis over which to normalize.

    Returns:
        Array with the same shape as ``x`` where values sum to one along ``axis``.
    """
    x = np.asarray(x, dtype=float)
    moved = np.moveaxis(x, axis, -1)
    max_vals = np.full(moved.shape[:-1], -np.inf, dtype=float)
    exp_sums = np.zeros(moved.shape[:-1], dtype=float)

    for index in range(moved.shape[-1]):
        current = moved[..., index]
        next_max = np.maximum(max_vals, current)
        exp_sums = exp_sums * np.exp(max_vals - next_max) + np.exp(current - next_max)
        max_vals = next_max

    normalized = np.exp(moved - np.expand_dims(max_vals, axis=-1)) / np.expand_dims(exp_sums, axis=-1)
    return np.moveaxis(normalized, -1, axis)


def layer_norm(x: FloatArray, eps: float = 1e-6) -> FloatArray:
    """Apply parameter-free layer normalization over the final axis."""
    x = np.asarray(x, dtype=float)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def welford_layer_norm(x: FloatArray, eps: float = 1e-6) -> FloatArray:
    """Apply layer normalization using Welford-style online moments.

    Args:
        x: Input with shape ``(..., dim)``.
        eps: Numerical stabilizer added to the variance.

    Returns:
        Normalized array with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")

    moved = np.moveaxis(x, -1, 0)
    count = 0
    mean = np.zeros(moved.shape[1:], dtype=float)
    m2 = np.zeros(moved.shape[1:], dtype=float)

    for feature in moved:
        count += 1
        delta = feature - mean
        mean = mean + delta / count
        delta2 = feature - mean
        m2 = m2 + (delta * delta2)

    variance = m2 / max(count, 1)
    normalized = (x - np.expand_dims(mean, axis=-1)) / np.sqrt(np.expand_dims(variance, axis=-1) + eps)
    return normalized


def feed_forward(
    x: Array2D | Array3D,
    w1: Array2D,
    b1: Array1D,
    w2: Array2D,
    b2: Array1D,
) -> Array2D | Array3D:
    """Apply a ReLU feed-forward block.

    Args:
        x: Input shaped ``(seq, dim)`` or ``(batch, seq, dim)``.
        w1: First projection shaped ``(hidden, dim)``.
        b1: First bias shaped ``(hidden,)``.
        w2: Second projection shaped ``(dim, hidden)``.
        b2: Second bias shaped ``(dim,)``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim not in (2, 3):
        raise ValueError("x must have shape (seq, dim) or (batch, seq, dim)")
    if w1.ndim != 2 or w2.ndim != 2 or b1.ndim != 1 or b2.ndim != 1:
        raise ValueError("feed-forward weights must be rank-1 or rank-2 arrays")
    if x.shape[-1] != w1.shape[1]:
        raise ValueError("x width must match w1 input width")
    if w1.shape[0] != b1.shape[0]:
        raise ValueError("w1 output width must match b1 width")
    if w2.shape[1] != w1.shape[0] or w2.shape[0] != b2.shape[0]:
        raise ValueError("w2 must map hidden width back to model width")
    hidden = np.maximum(0.0, (x @ w1.T) + b1)
    return (hidden @ w2.T) + b2


def glorot_scale(fan_in: int, fan_out: int) -> float:
    """Return the Glorot/Xavier standard-deviation scale."""
    return math.sqrt(2.0 / float(fan_in + fan_out))
