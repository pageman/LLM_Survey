"""Local optimizer-ablation dashboard demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.training import OptimizerAblationDashboard


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = OptimizerAblationDashboard().evaluate()
    report = build_report(
        experiment_id="optimizer_ablation_dashboard_demo",
        module="training.optimizer_ablation_dashboard",
        metrics={
            "variant_count": result["variant_count"],
            "best_loss": result["best_loss"],
            "loss_spread": result["loss_spread"],
            "best_stability_is_adamw": float(result["best_stability"] == "adamw"),
        },
        artifacts=result,
        notes=["Trajectory-level optimizer ablation comparing convergence and stability across variants."],
    )
    write_report(report, generated / "optimizer_ablation_dashboard_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
