"""Local structured prompting demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import StructuredPromptingDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = StructuredPromptingDemo().evaluate()
    report = build_report(
        experiment_id="structured_prompting_demo",
        module="utilization.structured_prompting_demo",
        metrics={
            "schema_gain": result["schema_gain"],
            "structured_success": result["structured_success"],
        },
        artifacts=result,
        notes=["Lite structured prompting demo comparing raw prompts against schema-guided prompts."],
    )
    write_report(report, output_dir / "structured_prompting_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
