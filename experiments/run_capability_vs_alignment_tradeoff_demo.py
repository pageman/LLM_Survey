"""Local capability-versus-alignment tradeoff demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.crosscutting import CapabilityAlignmentTradeoffDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = CapabilityAlignmentTradeoffDemo().evaluate()
    report = build_report(
        experiment_id="capability_vs_alignment_tradeoff_demo",
        module="capability_vs_alignment_tradeoff_demo",
        metrics={
            "integration_score": result["integration_score"],
            "tradeoff_correlation": result["tradeoff_correlation"],
        },
        artifacts=result,
        notes=["Capability-versus-alignment tradeoff demo with explicit frontier points and worst-alignment setting."],
    )
    write_report(report, output_dir / "capability_vs_alignment_tradeoff_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
