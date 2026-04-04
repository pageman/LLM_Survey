"""Program-aided reasoning demo with synthesis, execution, and reconciliation traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProgramAidedReasoningDemo:
    def evaluate(self) -> dict[str, object]:
        cases = [
            {
                "question": "What is 12 * 7?",
                "direct_score": 0.41,
                "program": "print(12 * 7)",
                "execution_result": "84",
                "program_score": 0.82,
                "reconciled_answer": "84",
            },
            {
                "question": "What is the average of 4, 10, and 16?",
                "direct_score": 0.44,
                "program": "nums=[4,10,16]; print(sum(nums)/len(nums))",
                "execution_result": "10.0",
                "program_score": 0.86,
                "reconciled_answer": "10",
            },
            {
                "question": "How many letters are in 'alignment'?",
                "direct_score": 0.39,
                "program": "print(len('alignment'))",
                "execution_result": "9",
                "program_score": 0.79,
                "reconciled_answer": "9",
            },
        ]
        direct = np.array([item["direct_score"] for item in cases], dtype=float)
        executed = np.array([item["program_score"] for item in cases], dtype=float)
        return {
            "direct": direct.tolist(),
            "program_aided": executed.tolist(),
            "execution_gain": float((executed - direct).mean()),
            "program_success": float(executed.mean()),
            "cases": cases,
        }
