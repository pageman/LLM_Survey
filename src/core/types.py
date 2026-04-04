"""Shared type aliases for NumPy-first educational modules."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Array1D = FloatArray
Array2D = FloatArray
Array3D = FloatArray
MaskArray = FloatArray

__all__ = [
    "Array1D",
    "Array2D",
    "Array3D",
    "FloatArray",
    "IntArray",
    "MaskArray",
]
