"""Toy data curriculum demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataCurriculumDemo:
    def evaluate(self) -> dict[str, object]:
        steps = [1, 2, 3, 4, 5]
        curriculum_scores = [0.55, 0.63, 0.71, 0.79, 0.84]
        shuffled_scores = [0.55, 0.60, 0.66, 0.71, 0.75]
        return {
            "steps": steps,
            "curriculum_scores": curriculum_scores,
            "shuffled_scores": shuffled_scores,
            "final_gain": curriculum_scores[-1] - shuffled_scores[-1],
        }
