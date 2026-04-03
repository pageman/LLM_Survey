"""Toy self-consistency reasoning demo with explicit sample aggregation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelfConsistencyDemo:
    def _sample_reasoning_paths(self) -> list[dict[str, object]]:
        return [
            {
                "path_id": 0,
                "reasoning_trace": ["identify operands", "apply arithmetic rule", "propose answer=42"],
                "answer": "42",
                "confidence": 0.81,
            },
            {
                "path_id": 1,
                "reasoning_trace": ["identify operands", "estimate outcome", "propose answer=42"],
                "answer": "42",
                "confidence": 0.74,
            },
            {
                "path_id": 2,
                "reasoning_trace": ["identify operands", "misapply subtraction", "propose answer=39"],
                "answer": "39",
                "confidence": 0.52,
            },
            {
                "path_id": 3,
                "reasoning_trace": ["identify operands", "recheck arithmetic", "propose answer=42"],
                "answer": "42",
                "confidence": 0.77,
            },
            {
                "path_id": 4,
                "reasoning_trace": ["identify operands", "shortcut pattern guess", "propose answer=41"],
                "answer": "41",
                "confidence": 0.49,
            },
        ]

    def evaluate(self) -> dict[str, object]:
        paths = self._sample_reasoning_paths()
        vote_counts: dict[str, int] = {}
        confidence_mass: dict[str, float] = {}
        for path in paths:
            answer = str(path["answer"])
            vote_counts[answer] = vote_counts.get(answer, 0) + 1
            confidence_mass[answer] = confidence_mass.get(answer, 0.0) + float(path["confidence"])

        single_path_score = float(paths[0]["confidence"])
        majority_answer = max(vote_counts, key=lambda answer: (vote_counts[answer], confidence_mass[answer]))
        majority_vote_score = (
            confidence_mass[majority_answer] / max(vote_counts[majority_answer], 1)
        ) + 0.04
        disagreement_rate = 1.0 - (vote_counts[majority_answer] / len(paths))

        return {
            "single_path_score": single_path_score,
            "majority_vote_score": majority_vote_score,
            "gain": majority_vote_score - single_path_score,
            "majority_answer": majority_answer,
            "vote_counts": vote_counts,
            "confidence_mass": confidence_mass,
            "disagreement_rate": disagreement_rate,
            "paths": paths,
        }
