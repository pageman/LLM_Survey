"""Multilingual prompting demo with prompt-format and language-drift structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultilingualPromptingDemo:
    def evaluate(self) -> dict[str, object]:
        prompt_rows = [
            {"language": "es", "prompt_format": "native", "score": 0.78},
            {"language": "de", "prompt_format": "native", "score": 0.75},
            {"language": "tl", "prompt_format": "native", "score": 0.72},
            {"language": "es", "prompt_format": "translated", "score": 0.70},
            {"language": "de", "prompt_format": "translated", "score": 0.67},
            {"language": "tl", "prompt_format": "translated", "score": 0.63},
        ]
        native = np.array([item["score"] for item in prompt_rows if item["prompt_format"] == "native"], dtype=float)
        translated = np.array(
            [item["score"] for item in prompt_rows if item["prompt_format"] == "translated"],
            dtype=float,
        )
        return {
            "native": native.tolist(),
            "translated": translated.tolist(),
            "native_prompt_score": float(native.mean()),
            "translation_gap": float((native - translated).mean()),
            "prompt_rows": prompt_rows,
        }
