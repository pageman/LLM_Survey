"""Local code-model architecture demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.architecture import CodeModelArchitectureDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = CodeModelArchitectureDemo().evaluate()
    report = build_report(
        experiment_id="code_model_architecture_demo",
        module="architecture.code_model_architecture_demo",
        metrics={"syntax_bias_gain": result["syntax_bias_gain"], "code_structure_score": result["code_structure_score"]},
        artifacts=result,
        notes=["Lite code-model architecture demo over token-only versus syntax-aware branches."],
    )
    write_report(report, output_dir / "code_model_architecture_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
