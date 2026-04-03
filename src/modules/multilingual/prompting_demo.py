"""Lite multilingual prompting demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultilingualPromptingDemo:
    def evaluate(self) -> dict[str, object]:
        native = np.array([0.78, 0.75, 0.72], dtype=float)
        translated = np.array([0.7, 0.67, 0.63], dtype=float)
        return {
            "native": native.tolist(),
            "translated": translated.tolist(),
            "native_prompt_score": float(native.mean()),
            "translation_gap": float((native - translated).mean()),
        }
