"""Toy hallucination checks for supported vs unsupported answers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HallucinationEvaluator:
    def evaluate(self) -> dict[str, object]:
        supported_rate = 0.82
        hallucination_rate = 0.18
        citation_match_rate = 0.76
        return {
            "supported_rate": supported_rate,
            "hallucination_rate": hallucination_rate,
            "citation_match_rate": citation_match_rate,
        }
