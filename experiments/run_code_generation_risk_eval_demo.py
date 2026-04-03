"""Local code-generation risk evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.crosscutting import CodeGenerationRiskEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = CodeGenerationRiskEvaluator().evaluate()
    report = build_report(
        experiment_id="code_generation_risk_eval_demo",
        module="code_generation_risk_eval",
        metrics={"risk_score": result["risk_score"], "max_risk": result["max_risk"]},
        artifacts=result,
        notes=["Lite code-generation risk evaluation over unsafe pattern rates."],
    )
    write_report(report, output_dir / "code_generation_risk_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
