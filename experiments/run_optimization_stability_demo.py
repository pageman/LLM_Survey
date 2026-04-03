from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import OptimizationStabilityDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = OptimizationStabilityDemo().evaluate()
    report = build_report("optimization_stability_demo", "systems.optimization_stability_demo", {"stability_gain": result["stability_gain"]}, result, ["Toy optimization stability and gradient clipping demo."])
    write_report(report, output_dir / "optimization_stability_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
