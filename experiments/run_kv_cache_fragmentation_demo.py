"""Local KV-cache fragmentation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import KVCacheFragmentationDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = KVCacheFragmentationDemo().evaluate()
    report = build_report(
        experiment_id="kv_cache_fragmentation_demo",
        module="systems.kv_cache_fragmentation_demo",
        metrics={
            "mean_fragmentation_penalty": result["mean_fragmentation_penalty"],
            "worst_case_penalty": result["worst_case_penalty"],
            "bucket_count": result["bucket_count"],
        },
        artifacts=result,
        notes=["KV-cache fragmentation demo with allocation-map rows across heterogeneous batch shapes."],
    )
    write_report(report, generated / "kv_cache_fragmentation_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
