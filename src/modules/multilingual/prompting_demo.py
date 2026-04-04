"""Multilingual prompting demo with prompt-format and language-drift structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np


class PromptRow(TypedDict):
    language: str
    prompt_format: str
    score: float
    drift_penalty: float


@dataclass
class MultilingualPromptingDemo:
    def evaluate(self) -> dict[str, object]:
        prompt_rows: list[PromptRow] = [
            {"language": "es", "prompt_format": "native", "score": 0.78, "drift_penalty": 0.03},
            {"language": "de", "prompt_format": "native", "score": 0.75, "drift_penalty": 0.04},
            {"language": "tl", "prompt_format": "native", "score": 0.72, "drift_penalty": 0.05},
            {"language": "es", "prompt_format": "translated", "score": 0.70, "drift_penalty": 0.08},
            {"language": "de", "prompt_format": "translated", "score": 0.67, "drift_penalty": 0.09},
            {"language": "tl", "prompt_format": "translated", "score": 0.63, "drift_penalty": 0.11},
        ]
        native = np.array([item["score"] for item in prompt_rows if item["prompt_format"] == "native"], dtype=float)
        translated = np.array(
            [item["score"] for item in prompt_rows if item["prompt_format"] == "translated"],
            dtype=float,
        )
        drift_penalty = np.array([item["drift_penalty"] for item in prompt_rows], dtype=float)
        return {
            "native": native.tolist(),
            "translated": translated.tolist(),
            "native_prompt_score": float(native.mean()),
            "translation_gap": float((native - translated).mean()),
            "mean_drift_penalty": float(drift_penalty.mean()),
            "prompt_rows": prompt_rows,
        }
