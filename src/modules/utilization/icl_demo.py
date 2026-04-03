"""Toy in-context learning demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ICLDemo:
    def evaluate(self) -> dict[str, object]:
        zero_shot = 0.58
        few_shot = 0.74
        return {
            "zero_shot_score": zero_shot,
            "few_shot_score": few_shot,
            "gain": few_shot - zero_shot,
        }
