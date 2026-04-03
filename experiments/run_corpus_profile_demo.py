"""Local corpus-profile demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.resources import CorpusProfileDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = CorpusProfileDemo().evaluate()
    report = build_report(
        experiment_id="corpus_profile_demo",
        module="resources.corpus_profile_demo",
        metrics={
            "domain_entropy": result["domain_entropy"],
            "code_fraction": result["code_fraction"],
            "multilingual_fraction": result["multilingual_fraction"],
        },
        artifacts=result,
        notes=["Dedicated corpus-profile summary over domain coverage and modality mix."],
    )
    write_report(report, generated / "corpus_profile_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
