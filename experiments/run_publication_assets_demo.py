"""Generate publication-facing Markdown and CSV assets."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.reporting.publication_assets import PublicationAssets


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = PublicationAssets(repo_root).build()
    report = build_report(
        experiment_id="publication_assets_demo",
        module="reporting.publication_assets",
        metrics={
            "module_count": result["module_count"],
            "benchmark_ranked_count": result["benchmark_ranked_count"],
            "benchmark_family_count": result["benchmark_family_count"],
            "benchmark_family_group_count": result["benchmark_family_group_count"],
            "mechanism_provenance_count": result["mechanism_provenance_count"],
            "survey_map_provenance_count": result["survey_map_provenance_count"],
            "stale_report_count": result["stale_report_count"],
        },
        artifacts=result["output_paths"],
        notes=["Generates publication-facing Markdown and CSV tables from canonical local reports."],
    )
    write_report(report, generated / "publication_assets_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
