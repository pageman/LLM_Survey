"""Local multilingual transfer evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.multilingual import TransferEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = TransferEvaluator().evaluate()
    report = build_report(
        experiment_id="transfer_eval_demo",
        module="multilingual.transfer_eval",
        metrics={"transfer_score": result["transfer_score"], "few_shot_gain": result["few_shot_gain"]},
        artifacts=result,
        notes=["Lite multilingual transfer demo over zero-shot and few-shot settings."],
    )
    write_report(report, output_dir / "transfer_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
