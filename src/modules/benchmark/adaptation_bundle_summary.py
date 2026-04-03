"""Dedicated adaptation-bundle summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AdaptationBundleSummary:
    reports_dir: str | Path

    def build(self) -> dict[str, object]:
        reports_dir = Path(self.reports_dir)
        filenames = [
            "finetuning_demo.json",
            "instruction_tuning_demo.json",
            "peft_lora_demo.json",
            "dpo_toy_demo.json",
            "ppo_rlhf_toy_demo.json",
        ]
        rows = {}
        gains = []
        for filename in filenames:
            payload = json.loads((reports_dir / filename).read_text())
            gain = float(payload["metrics"]["gain"])
            rows[filename.replace("_demo.json", "")] = round(gain, 4)
            gains.append(gain)
        return {
            "num_reports": len(filenames),
            "mean_gain": round(sum(gains) / max(len(gains), 1), 4),
            "best_gain": round(max(gains), 4),
            "bundle_rows": rows,
        }
