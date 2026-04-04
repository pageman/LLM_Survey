"""Local jailbreak-transfer evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import JailbreakTransferEvaluator


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = JailbreakTransferEvaluator().evaluate()
    report = build_report(
        experiment_id="jailbreak_transfer_eval_demo",
        module="evaluation.jailbreak_transfer_eval",
        metrics={
            "source_attack_rate": result["source_attack_rate"],
            "transfer_attack_rate": result["transfer_attack_rate"],
            "transfer_ratio": result["transfer_ratio"],
            "attack_family_count": result["attack_family_count"],
        },
        artifacts=result,
        notes=["Jailbreak-transfer evaluation with attack-family transfer rows and defense-gap accounting."],
    )
    write_report(report, generated / "jailbreak_transfer_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
