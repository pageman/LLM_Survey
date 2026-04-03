"""Local DPO-style adaptation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation.dpo_toy import DPOToyExperiment


def main() -> None:
    output_dir = Path("artifacts/generated")
    preferences = [
        ("answer safely", "provide a cautious response", "give dangerous instructions"),
        ("decline harmful request", "refuse and explain safety", "comply directly"),
        ("summarize neutrally", "give a balanced summary", "write a biased rant"),
    ]
    experiment = DPOToyExperiment()
    result = experiment.adapt(preferences=preferences, eval_preference=preferences[0], epochs=14)
    report = build_report(
        experiment_id="dpo_toy_demo",
        module="adaptation.dpo_toy",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
            "trainable_fraction": result["trainable_fraction"],
        },
        artifacts=result,
        notes=["Lite DPO-style preference optimization with explicit chosen-vs-rejected margin updates."],
    )
    write_report(report, output_dir / "dpo_toy_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
