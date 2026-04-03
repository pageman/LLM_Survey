"""Toy prefix-decoder demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrefixDecoderDemo:
    def evaluate(self, prefix: str, continuation: str) -> dict[str, object]:
        combined = f"{prefix} {continuation}".strip()
        return {
            "prefix": prefix,
            "continuation": continuation,
            "combined": combined,
            "prefix_length": len(prefix.split()),
        }
