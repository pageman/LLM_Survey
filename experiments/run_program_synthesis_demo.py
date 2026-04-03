"""Local program-synthesis code-pretraining demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.code_pretraining import ProgramSynthesisDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ProgramSynthesisDemo().evaluate()
    report = build_report(
        experiment_id="program_synthesis_demo",
        module="code_pretraining.program_synthesis_demo",
        metrics={"exact_match": result["exact_match"], "execution_success": result["execution_success"]},
        artifacts=result,
        notes=["Lite program synthesis demo over spec satisfaction and execution success."],
    )
    write_report(report, output_dir / "program_synthesis_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
