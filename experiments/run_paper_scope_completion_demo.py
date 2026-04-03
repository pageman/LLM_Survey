"""Generate paper-scope completion reports for the remaining tracked targets."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation.paper_scope_completion import (
    REMAINING_PAPER_SCOPE_MODULES,
    PaperScopeCompletionGenerator,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    generator = PaperScopeCompletionGenerator()
    emitted = []

    for module_name in REMAINING_PAPER_SCOPE_MODULES:
        payload = generator.build_payload(module_name)
        experiment_id = module_name.replace(".", "_")
        report = build_report(
            experiment_id=experiment_id,
            module=module_name,
            metrics=payload["metrics"],
            artifacts=payload["artifacts"],
            notes=payload["notes"],
        )
        write_report(report, generated / f"{experiment_id}.json")
        emitted.append(experiment_id)

    summary = build_report(
        experiment_id="paper_scope_completion_demo",
        module="evaluation.paper_scope_completion",
        metrics={
            "num_generated_reports": len(emitted),
            "target_modules": len(REMAINING_PAPER_SCOPE_MODULES),
        },
        artifacts={"generated_experiment_ids": emitted},
        notes=["Generates explicit lite paper-scope implementation reports for the remaining tracked modules."],
    )
    write_report(summary, generated / "paper_scope_completion_demo.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
