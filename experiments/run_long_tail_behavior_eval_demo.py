"""Local long-tail behavior evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import LongTailBehaviorEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = LongTailBehaviorEvaluator().evaluate()
    report = build_report(
        experiment_id="long_tail_behavior_eval_demo",
        module="evaluation.long_tail_behavior_eval",
        metrics={"head_tail_gap": result["head_tail_gap"], "tail_score": result["tail_score"]},
        artifacts=result,
        notes=["Lite long-tail behavior evaluation over head versus tail cases."],
    )
    write_report(report, output_dir / "long_tail_behavior_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
