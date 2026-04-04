"""Lite verifier-guided reasoning evaluation with acceptance decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VerifierEvaluator:
    acceptance_threshold: float = 0.7

    def evaluate(self) -> dict[str, object]:
        proposals = [
            {"question": "2 + 2", "candidate": "4", "base_score": 0.61, "verifier_score": 0.91, "correct": True},
            {"question": "capital of france", "candidate": "Paris", "base_score": 0.58, "verifier_score": 0.89, "correct": True},
            {"question": "prime after 11", "candidate": "12", "base_score": 0.54, "verifier_score": 0.33, "correct": False},
            {"question": "water formula", "candidate": "H2O", "base_score": 0.6, "verifier_score": 0.86, "correct": True},
        ]
        base_scores = np.array([item["base_score"] for item in proposals], dtype=float)
        verified_scores = np.array([item["verifier_score"] for item in proposals], dtype=float)
        accepted = [item for item in proposals if item["verifier_score"] >= self.acceptance_threshold]
        false_accepts = sum(1 for item in accepted if not item["correct"])
        false_rejects = sum(1 for item in proposals if item["verifier_score"] < self.acceptance_threshold and item["correct"])
        return {
            "verifier_gain": float((verified_scores - base_scores).mean()),
            "verified_score": float(verified_scores.mean()),
            "acceptance_rate": len(accepted) / len(proposals),
            "false_accept_rate": false_accepts / len(proposals),
            "false_reject_rate": false_rejects / len(proposals),
            "threshold": self.acceptance_threshold,
            "proposals": proposals,
        }
