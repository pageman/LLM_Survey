"""Local PPO-RLHF adaptation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation import PPORLFHToy


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = PPORLFHToy().evaluate()
    report = build_report(
        experiment_id="ppo_rlhf_toy_demo",
        module="adaptation.ppo_rlhf_toy",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
            "acceptance_rate": result["acceptance_rate"],
        },
        artifacts=result,
        notes=["PPO-style RLHF toy demo with rollout traces, clipping, and KL-regularized policy updates."],
    )
    write_report(report, output_dir / "ppo_rlhf_toy_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
