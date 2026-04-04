"""Code-eval style demo with syntax, semantics, and repairability metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CodeEvalDemo:
    def evaluate(self) -> dict[str, object]:
        tasks = [
            {"task": "two_sum", "syntax_valid": 0.96, "semantic_correct": 0.72, "repairable": 0.83},
            {"task": "fizz_buzz", "syntax_valid": 0.94, "semantic_correct": 0.81, "repairable": 0.89},
            {"task": "balanced_parentheses", "syntax_valid": 0.91, "semantic_correct": 0.66, "repairable": 0.78},
        ]
        pass_at_k = np.array([0.31, 0.52, 0.68], dtype=float)
        syntax_validity = np.array([item["syntax_valid"] for item in tasks], dtype=float)
        semantic_correctness = np.array([item["semantic_correct"] for item in tasks], dtype=float)
        repairability = np.array([item["repairable"] for item in tasks], dtype=float)
        return {
            "k_values": [1, 5, 10],
            "pass_at_k": pass_at_k.tolist(),
            "pass_at_1": float(pass_at_k[0]),
            "pass_at_10": float(pass_at_k[-1]),
            "syntax_validity": float(syntax_validity.mean()),
            "semantic_correctness": float(semantic_correctness.mean()),
            "repairability": float(repairability.mean()),
            "tasks": tasks,
        }
