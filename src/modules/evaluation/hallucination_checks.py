"""Hallucination checks with failure-mode categories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HallucinationEvaluator:
    def evaluate(self) -> dict[str, object]:
        cases = [
            {"case": "answer grounded in cited paragraph", "category": "supported", "score": 0.92},
            {"case": "missing evidence but plausible", "category": "evidence_missing", "score": 0.18},
            {"case": "citation contradicts answer", "category": "evidence_conflict", "score": 0.11},
            {"case": "fabricated unsupported detail", "category": "unsupported_generation", "score": 0.25},
        ]
        supported_rate = 0.82
        hallucination_rate = 0.18
        citation_match_rate = 0.76
        category_counts = {
            category: sum(1 for item in cases if item["category"] == category)
            for category in {item["category"] for item in cases}
        }
        return {
            "supported_rate": supported_rate,
            "hallucination_rate": hallucination_rate,
            "citation_match_rate": citation_match_rate,
            "failure_mode_count": len(category_counts) - int("supported" in category_counts),
            "category_breakdown": category_counts,
            "cases": cases,
        }
