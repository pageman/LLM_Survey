"""Local safety-versus-reasoning tradeoff demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.crosscutting import SafetyReasoningTradeoffDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = SafetyReasoningTradeoffDemo().evaluate()
    report = build_report(
        experiment_id="safety_reasoning_tradeoff_demo",
        module="safety_reasoning_tradeoff_demo",
        metrics={"risk_score": result["risk_score"], "tradeoff_correlation": result["tradeoff_correlation"]},
        artifacts=result,
        notes=["Lite safety-versus-reasoning tradeoff demo over capability and safety trends."],
    )
    write_report(report, output_dir / "safety_reasoning_tradeoff_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
