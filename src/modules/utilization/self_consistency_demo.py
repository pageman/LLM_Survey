"""Toy self-consistency reasoning demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelfConsistencyDemo:
    def evaluate(self) -> dict[str, object]:
        single_path = 0.72
        majority_vote = 0.81
        return {
            "single_path_score": single_path,
            "majority_vote_score": majority_vote,
            "gain": majority_vote - single_path,
        }
