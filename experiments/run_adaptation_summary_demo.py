"""Build an adaptation-only summary view from generated reports."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import AdaptationSummary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    summary = AdaptationSummary(generated).build()

    best_gain = summary["ranked_by_gain"][0]["gain"] if summary["ranked_by_gain"] else None
    best_loss = summary["ranked_by_adapted_loss"][0]["adapted_loss"] if summary["ranked_by_adapted_loss"] else None

    report = build_report(
        experiment_id="adaptation_summary_demo",
        module="evaluation.adaptation_summary",
        metrics={
            "num_adaptation_reports": summary["num_adaptation_reports"],
            "best_gain": best_gain,
            "best_adapted_loss": best_loss,
        },
        artifacts=summary,
        notes=["Adaptation-only ranking view built without mixing in retrieval or long-context reports."],
    )
    write_report(report, generated / "adaptation_summary_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
