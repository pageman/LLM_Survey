"""Local domain-coverage pretraining demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import DomainCoverageDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = DomainCoverageDemo().evaluate()
    report = build_report(
        experiment_id="domain_coverage_demo",
        module="pretraining.domain_coverage_demo",
        metrics={
            "tail_coverage": result["tail_coverage"],
            "domain_entropy": result["domain_entropy"],
            "cross_domain_gap": result["cross_domain_gap"],
        },
        artifacts=result,
        notes=["Domain-coverage demo with head-tail mixture mass and held-out transfer gaps."],
    )
    write_report(report, output_dir / "domain_coverage_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
