from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import KVCacheToy


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = KVCacheToy().evaluate()
    report = build_report("kv_cache_toy_demo", "systems.kv_cache_toy", {"speedup": result["speedup"]}, result, ["Toy autoregressive KV-cache cost reduction demo."])
    write_report(report, output_dir / "kv_cache_toy_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
