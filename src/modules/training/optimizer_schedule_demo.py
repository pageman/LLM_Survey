"""Lite optimizer-schedule demo with warmup and decay curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OptimizerScheduleDemo:
    def evaluate(self) -> dict[str, object]:
        steps = np.arange(1, 9, dtype=float)
        schedule = np.minimum(steps / 3.0, 1.0) * (steps ** -0.5)
        loss = np.array([1.38, 1.22, 1.12, 1.06, 1.02, 0.99, 0.97, 0.96], dtype=float)
        return {
            "schedule": schedule.round(6).tolist(),
            "peak_step": int(np.argmax(schedule) + 1),
            "final_loss": float(loss[-1]),
            "schedule_gain": float(loss[0] - loss[-1]),
            "loss_curve": loss.tolist(),
        }
