from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import PipelineParallelismDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = PipelineParallelismDemo().evaluate()
    report = build_report("pipeline_parallelism_demo", "systems.pipeline_parallelism", {"throughput_gain": result["throughput_gain"]}, result, ["Toy GPipe-style throughput demo."])
    write_report(report, output_dir / "pipeline_parallelism_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
