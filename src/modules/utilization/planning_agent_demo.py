"""Toy planning/agent utilization demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlanningAgentDemo:
    def evaluate(self) -> dict[str, object]:
        plan_steps = ["understand goal", "retrieve facts", "draft answer", "verify answer"]
        success_rate = 0.78
        return {
            "plan_steps": plan_steps,
            "success_rate": success_rate,
            "num_steps": len(plan_steps),
        }
