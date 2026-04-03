"""Local multilingual architecture demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.architecture import MultilingualArchitectureDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = MultilingualArchitectureDemo().evaluate()
    report = build_report(
        experiment_id="multilingual_architecture_demo",
        module="architecture.multilingual_architecture_demo",
        metrics={"parameter_sharing_score": result["parameter_sharing_score"], "transfer_score": result["transfer_score"]},
        artifacts=result,
        notes=["Lite multilingual architecture demo over shared-parameter transfer behavior."],
    )
    write_report(report, output_dir / "multilingual_architecture_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
