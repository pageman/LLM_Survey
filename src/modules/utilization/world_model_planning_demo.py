"""World-model planning demo with explicit state transitions and replanning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WorldModelPlanningDemo:
    def evaluate(self) -> dict[str, object]:
        rollout = [
            {
                "step": 0,
                "latent_state": "observe_room",
                "candidate_actions": ["scan_doors", "move_forward"],
                "chosen_action": "scan_doors",
                "predicted_next_value": 0.21,
                "realized_value": 0.25,
                "replanned": False,
            },
            {
                "step": 1,
                "latent_state": "mapped_exit_candidates",
                "candidate_actions": ["open_left", "open_right"],
                "chosen_action": "open_right",
                "predicted_next_value": 0.47,
                "realized_value": 0.44,
                "replanned": True,
            },
            {
                "step": 2,
                "latent_state": "corridor_progress",
                "candidate_actions": ["continue", "inspect_side_path"],
                "chosen_action": "continue",
                "predicted_next_value": 0.69,
                "realized_value": 0.71,
                "replanned": False,
            },
            {
                "step": 3,
                "latent_state": "goal_visible",
                "candidate_actions": ["exit", "search_bonus"],
                "chosen_action": "exit",
                "predicted_next_value": 1.0,
                "realized_value": 1.0,
                "replanned": False,
            },
        ]
        state_values = np.array([item["realized_value"] for item in rollout], dtype=float)
        prediction_error = np.array(
            [abs(item["realized_value"] - item["predicted_next_value"]) for item in rollout],
            dtype=float,
        )
        return {
            "state_values": state_values.tolist(),
            "plan_success": float(state_values[-1]),
            "state_value_gain": float(state_values[-1] - state_values[0]),
            "replanning_rate": float(np.mean([item["replanned"] for item in rollout])),
            "mean_prediction_error": float(prediction_error.mean()),
            "rollout": rollout,
        }
