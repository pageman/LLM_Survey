"""Toy embodied-agent application stub."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmbodiedAgentStub:
    def evaluate(self) -> dict[str, object]:
        actions = ["locate mug", "reach mug", "grasp mug", "place mug on table"]
        task_success_rate = 0.73
        return {
            "actions": actions,
            "task_success_rate": task_success_rate,
        }
