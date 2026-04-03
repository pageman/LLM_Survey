"""Dedicated risk-bundle summary over generated evaluation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RiskBundleSummary:
    reports_dir: str | Path

    def evaluate(self) -> dict[str, object]:
        reports_dir = Path(self.reports_dir)
        targets = {
            "hallucination_checks_demo.json": ("hallucination_rate", "min"),
            "safety_eval_demo.json": ("jailbreak_success_rate", "min"),
            "bias_eval_demo.json": ("stereotype_score", "min"),
            "truthfulness_eval_demo.json": ("truthfulness_score", "max"),
            "privacy_leakage_eval_demo.json": ("privacy_risk", "min"),
        }
        scores = {}
        for filename, (metric, goal) in targets.items():
            data = json.loads((reports_dir / filename).read_text())
            value = float(data["metrics"][metric])
            normalized = value if goal == "max" else 1.0 - value
            scores[filename.replace("_demo.json", "")] = round(normalized, 4)
        bundle_score = sum(scores.values()) / max(len(scores), 1)
        return {
            "component_scores": scores,
            "bundle_score": round(bundle_score, 4),
            "risk_floor": round(min(scores.values()), 4),
        }
