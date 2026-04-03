from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import ICLDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ICLDemo().evaluate()
    report = build_report("icl_demo", "utilization.icl_demo", {"gain": result["gain"]}, result, ["Toy in-context learning gain demo."])
    write_report(report, output_dir / "icl_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
