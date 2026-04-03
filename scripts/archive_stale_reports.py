"""Archive stale duplicate report artifacts and optionally prune them from the active directory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive stale duplicate report artifacts.")
    parser.add_argument("--repo-root", default="/Users/hifi/Downloads/LLM_Survey")
    parser.add_argument("--execute", action="store_true", help="Move stale reports into an archive folder.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    generated = repo_root / "artifacts" / "generated"
    archive_root = repo_root / "artifacts" / "archive" / "stale_reports"
    index = json.loads((generated / "report_index_demo.json").read_text())
    stale_reports = index["artifacts"].get("stale_reports", [])

    print(f"stale_report_count={len(stale_reports)}")
    for row in stale_reports:
        print(row["path"])

    if not args.execute:
        print("dry_run_only=true")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = archive_root / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in stale_reports:
        source = Path(row["path"])
        if source.exists():
            destination = archive_dir / source.name
            shutil.move(str(source), str(destination))
            manifest.append(
                {
                    "experiment_id": row["experiment_id"],
                    "module": row["module"],
                    "from": str(source),
                    "to": str(destination),
                }
            )
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"archived_to={archive_dir}")
    print("pruned_from_active_dir=true")


if __name__ == "__main__":
    main()
