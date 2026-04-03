"""Generate a human-readable Markdown scoreboard from local report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import DocsSummaryGenerator


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    generator = DocsSummaryGenerator(generated)
    markdown = generator.build_markdown()
    scoreboard_path = docs_dir / "scoreboard.md"
    scoreboard_path.write_text(markdown)

    progress = generator.compute_progress()
    report = build_report(
        experiment_id="docs_summary_demo",
        module="evaluation.docs_summary",
        metrics={
            "implemented": progress["implemented"],
            "target": progress["target"],
            "percentage": progress["percentage"],
        },
        artifacts={"scoreboard_path": str(scoreboard_path)},
        notes=["Generates a human-readable Markdown scoreboard from the local report artifacts."],
    )
    write_report(report, generated / "docs_summary_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
