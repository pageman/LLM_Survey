"""Local risk-bundle summary demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import RiskBundleSummary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = RiskBundleSummary(generated).evaluate()
    report = build_report(
        experiment_id="risk_bundle_summary_demo",
        module="benchmark.risk_bundle_summary",
        metrics={"bundle_score": result["bundle_score"], "risk_floor": result["risk_floor"]},
        artifacts=result,
        notes=["Dedicated risk-bundle summary over core evaluation artifacts."],
    )
    write_report(report, generated / "risk_bundle_summary_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
