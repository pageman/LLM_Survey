"""Protocol interfaces for reusable forward-style modules."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Array2D, MaskArray


@runtime_checkable
class ForwardModule(Protocol):
    """Protocol for modules exposing a sequence-major forward pass."""

    def forward(self, x: Array2D, mask: MaskArray | None = None) -> Array2D:
        """Return the forward result for one sequence-major input."""


@runtime_checkable
class AttentionModule(Protocol):
    """Protocol for attention-like modules over query/key/value inputs."""

    def forward(
        self,
        query: Array2D,
        key: Array2D,
        value: Array2D,
        mask: MaskArray | None = None,
    ) -> Array2D:
        """Return the attention output for sequence-major inputs."""


__all__ = ["AttentionModule", "ForwardModule"]
