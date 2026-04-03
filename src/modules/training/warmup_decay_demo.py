"""Lite warmup-decay schedule demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WarmupDecayDemo:
    def evaluate(self) -> dict[str, object]:
        steps = np.arange(1, 10, dtype=float)
        schedule = np.minimum(steps / 3.0, 1.0) / np.sqrt(steps)
        return {
            "schedule": schedule.round(6).tolist(),
            "peak_step": int(np.argmax(schedule) + 1),
            "stability_score": float(1.0 - np.std(np.diff(schedule))),
        }
