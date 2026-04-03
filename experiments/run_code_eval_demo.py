"""Local code-eval capability demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import CodeEvalDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = CodeEvalDemo().evaluate()
    report = build_report(
        experiment_id="code_eval_demo",
        module="evaluation.code_eval_demo",
        metrics={"pass_at_1": result["pass_at_1"], "pass_at_10": result["pass_at_10"]},
        artifacts=result,
        notes=["Lite code-eval style capability demo with pass@k metrics."],
    )
    write_report(report, output_dir / "code_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
