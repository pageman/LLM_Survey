"""Build a simple adaptation leaderboard artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import AdaptationLeaderboard


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    leaderboard = AdaptationLeaderboard(generated).build()

    best_gain = leaderboard["top_by_gain"][0]["gain"] if leaderboard["top_by_gain"] else None
    best_efficiency = (
        leaderboard["top_by_efficiency"][0]["efficiency_score"]
        if leaderboard["top_by_efficiency"]
        else None
    )
    best_loss = (
        leaderboard["top_by_lowest_adapted_loss"][0]["adapted_loss"]
        if leaderboard["top_by_lowest_adapted_loss"]
        else None
    )

    report = build_report(
        experiment_id="adaptation_leaderboard_demo",
        module="evaluation.adaptation_leaderboard",
        metrics={
            "num_ranked": leaderboard["num_ranked"],
            "best_gain": best_gain,
            "best_efficiency_score": best_efficiency,
            "best_adapted_loss": best_loss,
        },
        artifacts=leaderboard,
        notes=["Presentation-only leaderboard layered on top of adaptation summary without changing report schema."],
    )
    write_report(report, generated / "adaptation_leaderboard_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
