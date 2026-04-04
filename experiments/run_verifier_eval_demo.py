"""Local verifier-guided reasoning evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import VerifierEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = VerifierEvaluator().evaluate()
    report = build_report(
        experiment_id="verifier_eval_demo",
        module="evaluation.verifier_eval",
        metrics={
            "verifier_gain": result["verifier_gain"],
            "verified_score": result["verified_score"],
            "acceptance_rate": result["acceptance_rate"],
            "false_accept_rate": result["false_accept_rate"],
        },
        artifacts=result,
        notes=["Verifier-guided reasoning demo with proposal-level acceptance, thresholding, and error accounting."],
    )
    write_report(report, output_dir / "verifier_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
