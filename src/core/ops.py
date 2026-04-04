"""Shared NumPy ops used across attention and transformer primitives."""

from __future__ import annotations

import math

import numpy as np


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return a numerically stable softmax over the given axis.

    Args:
        x: Input array.
        axis: Axis over which to normalize.

    Returns:
        Array with the same shape as ``x`` where values sum to one along ``axis``.
    """
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply parameter-free layer normalization over the final axis."""
    x = np.asarray(x, dtype=float)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def feed_forward(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Apply a ReLU feed-forward block."""
    x = np.asarray(x, dtype=float)
    hidden = np.maximum(0.0, (x @ w1.T) + b1)
    return (hidden @ w2.T) + b2


def glorot_scale(fan_in: int, fan_out: int) -> float:
    """Return the Glorot/Xavier standard-deviation scale."""
    return math.sqrt(2.0 / float(fan_in + fan_out))
